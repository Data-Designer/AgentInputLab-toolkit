"""Credential-free planning and explicitly authorized asset preparation."""

from __future__ import annotations

import itertools
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from agentinputlab.api import describe_intervention
from agentinputlab.benchmarks import BenchmarkRegistry, describe_benchmark
from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import to_jsonable
from agentinputlab.experiments.secrets import environment_secret_values, redact_secret_text
from agentinputlab.registry import AlgorithmRegistry

from .configuration import WorkflowConfig, file_sha, native_configuration
from .readiness import (
    budget_and_input_readiness,
    local_references,
    method_readiness,
    native_arena_readiness,
    predictor_readiness,
)
from .recipes import complete_configuration, recipe

_PROMPT_ONLY = {"baseline.no_op", "plan.plan_and_solve.ps_prompt", "plan.plan_and_solve.ps_plus_prompt"}
_ARENA_CELLS = {"tooltalk": "tooltalk.full", "statebench": "state_bench.common_support",
                "toolsandbox": "toolsandbox.common_support", "tau_banking": "tau3.banking_knowledge",
                "tau_retail": "tau3.retail"}


def _beir_workflow_source(workflow):
    """Bind one archive owner and one split for inspection, preparation and execution."""
    from agentinputlab.benchmarks.beir_archive import BEIR_ARCHIVE_LOCKS, validate_split

    target, configured = workflow.target, workflow.run.retriever
    cell = target["benchmark_id"]
    owns_source = configured.get("kind") == "beir_archive"
    if owns_source:
        if cell != configured.get("benchmark_id") or any(key in target for key in (
            "prepared_root", "cache_dir", "beir_sample_sources", "trec_sample_sources", "longbench_sources",
        )):
            raise ValidationError("BEIR retriever owns its matching archive; do not supply a second dataset source")
        if configured.get("split", target["split"]) != target["split"]:
            raise ValidationError("BEIR retriever split must match the workflow split")
    if owns_source or (cell in BEIR_ARCHIVE_LOCKS and not any(key in target for key in (
        "prepared_root", "beir_sample_sources", "trec_sample_sources",
    ))):
        validate_split(cell, target["split"])
    return configured if owns_source else target


def safe_error(error, workflow):
    names = (workflow.run.backend.get("api_key_env", "OPENAI_API_KEY"),)
    values = environment_secret_values(os.environ, names)
    return {"type": type(error).__name__, "message": redact_secret_text(str(error), secret_values=values)}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValidationError("endpoint readiness probe does not follow redirects")


def _probe_endpoint(workflow):
    """One optional GET /models; no generation or schema-success claim."""
    backend = workflow.run.backend
    root = backend.get("base_url", "")
    parsed = urllib.parse.urlsplit(root)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValidationError("endpoint probe requires an HTTP(S) API root without user information")
    if parsed.query or parsed.fragment:
        raise ValidationError("endpoint probe API root must not contain a query or fragment")
    key = os.environ.get(backend.get("api_key_env", "OPENAI_API_KEY"))
    headers = {"Accept": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    request = urllib.request.Request(root.rstrip("/") + "/models", headers=headers)
    try:
        with urllib.request.build_opener(_NoRedirect()).open(request, timeout=10) as response:
            raw = response.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024:
                raise ValidationError("endpoint models response exceeds 1 MiB")
            value = json.loads(raw)
    except urllib.error.HTTPError as error:
        if error.code in {404, 405, 501}:
            return {"status": "ready", "http_status": error.code, "configured_model_listed": None,
                    "model_listing_supported": False, "generation_protocol_verified": False,
                    "generation_calls": 0,
                    "action": "Endpoint responds but does not expose model listing; the first requested case verifies generation."}
        return {"status": "blocked", "http_status": error.code,
                "action": "Check API root, authorization and provider model-list support."}
    if not isinstance(value, dict) or not isinstance(value.get("data"), list):
        raise ValidationError("endpoint did not return an OpenAI-compatible model list")
    found = any(isinstance(item, dict) and item.get("id") == backend.get("model_id") for item in value["data"])
    return {"status": "ready" if found else "blocked", "configured_model_listed": found,
            "generation_protocol_verified": False, "generation_calls": 0,
            "action": None if found else "Choose the exact served model ID or inspect the provider's routing."}


def inspect_workflow(workflow: WorkflowConfig, *, probe_endpoint=False, _inspect_data=True, _adapters=None):
    """Inspect prerequisites without downloading, fitting, or generating text.

    ``can_attempt`` means the inspected gates pass, not that every native
    runtime or model output has been verified. Unknown checks stay unknown.
    """
    if type(probe_endpoint) is not bool:
        raise ValidationError("probe_endpoint must be boolean")
    requested_id = workflow.identity
    checks = []
    if workflow.max_workers != 1:
        checks.append({"check": "scheduling", "status": "blocked", "action":
                       "This first workflow wrapper is sequential. Existing campaign/Studio concurrency is unchanged; use max_workers=1 here."})

    def check(name, operation, action):
        try:
            details = operation()
            checks.append({"check": name, "status": "ready", "details": details})
        except Exception as error:
            checks.append({"check": name, "status": "blocked", "error": safe_error(error, workflow),
                           "action": action})

    from .artifacts import discover_artifacts
    try:
        workflow, discovery = discover_artifacts(workflow)
        checks.append({"check": "artifact_discovery", "status": "ready", "details": discovery})
    except Exception as error:
        checks.append({"check": "artifact_discovery", "status": "blocked", "error": safe_error(error, workflow),
                       "action": "Provide an unambiguous catalog entry for this exact model revision and benchmark, or bind artifacts explicitly."})
    check("configuration", workflow.run.validate, "Complete native parameters with `agentinputlab show RELEASE --parameters`.")
    if checks[-1]["status"] == "ready" and "router" not in workflow.run.artifacts:
        def constructor():
            contract = recipe(workflow.run.release_id)
            if contract["route"] in {"weight_training", "context_preparation", "unavailable"}:
                return {"handled_by_native_recipe": True}
            missing = [value["path"] for value in workflow.run.artifacts.values()
                       if isinstance(value, Mapping) and value.get("path") and not Path(value["path"]).exists()]
            if missing:
                planned = {asset["destination"] for asset in workflow.preparation.get("assets", ())}
                if not all(path in planned for path in missing):
                    raise ValidationError("required artifact is missing; bind an existing artifact or declare its pinned preparation asset")
                return {"artifact_contents_checked": False, "awaiting_pinned_asset_preparation": True, "model_calls": 0}
            method = workflow.run.create_intervention()
            return {"constructed": True, "class": type(method).__name__, "model_calls": 0}
        check("native_parameters", constructor,
              "Complete the method's native semantic fields shown in this error before preparing or contacting a model.")
    check("bound_inputs", workflow.verify_bindings, "Restore the exact bound inputs or prepare a new workflow.")
    metadata = None
    try:
        metadata = describe_intervention(workflow.run.release_id)
    except Exception as error:
        checks.append({"check": "profile", "status": "blocked", "error": safe_error(error, workflow),
                       "action": "Install the declared optional runtime and inspect the selected profile."})
    if metadata and (metadata.get("blocker") or metadata.get("execution_kind") == "training_recipe"):
        contract = recipe(workflow.run.release_id)
        checks.append({"check": "profile", "status": "blocked", "action":
                       ("Use the native context/skill preparation configuration and bind its output artifact for evaluation."
                        if contract["route"] == "context_preparation" else
                        "Use the explicit weight-training recipe or an already trained inference profile."),
                       "route": contract["route"]})
    needs_backend = workflow.target["kind"] != "query" or workflow.run.release_id not in _PROMPT_ONLY
    predictor = workflow.target.get("predictor_parameters", {})
    if (workflow.target["kind"] == "benchmark"
            and workflow.target["benchmark_id"] == "beir.fiqa_2018"
            and workflow.run.release_id == "baseline.no_op"
            and workflow.run.retriever.get("kind") == "beir_archive"):
        needs_backend = False  # the native query/ranking control does not generate text
    if (workflow.run.release_id == "acquire.qrag.common_support"
            and "trace_ranking" in predictor):
        needs_backend = False  # predictor readiness still validates the exact protocol and corpus
    if "archived_retrieval" in predictor:
        needs_backend = False
        if workflow.run.backend:
            checks.append({"check": "predictor_inputs", "status": "blocked",
                           "action": "Archived ranking is score-only; remove the live backend instead of implicitly calling it."})
    if needs_backend and not workflow.run.backend:
        checks.append({"check": "backend", "status": "blocked", "action":
                       "Supply an actual model_id, API root and deployment revision in run.backend."})
    if workflow.required_capabilities:
        def capabilities():
            from agentinputlab.backends.llm import ModelCapability, require_capabilities
            backend = workflow.run._backend()
            if backend is None:
                raise ValidationError("required capabilities need a backend")
            require_capabilities(backend, (ModelCapability(x) for x in workflow.required_capabilities))
            return {"basis": "operator-declared capabilities", "generation_verified": False}
        check("model_capabilities", capabilities, "Use a capable endpoint; do not claim a capability solely to pass this check.")
    check("method_requirements", lambda: method_readiness(workflow, dependencies=False),
          "Provide the exact profile's native runtime, retrieval configuration or prepared artifact; see recipe details.")
    check("worker_environment", lambda: method_readiness(workflow),
          "Prepare the selected worker interpreter, dependencies and artifact files.")
    check("local_assets", lambda: local_references(workflow.run),
          "Prepare the missing bound sources/assets before model execution.")
    assets = []
    for name, value in workflow.run.runtime.items():
        if isinstance(value, str) and (name.endswith("_path") or name in {
                "checkout", "source_root", "python_executable", "checkpoint_root", "embedder_root",
                "artifact_root", "base_model_root", "feature_model_root", "tokenizer_root", "nltk_data"}):
            assets.append(("runtime." + name, value, None))
    for name, reference in workflow.run.artifacts.items():
        if isinstance(reference, Mapping) and reference.get("path"):
            assets.append(("artifacts." + name, reference["path"], reference.get("sha256")))
    for name, path, expected in assets:
        def verify(path=path, expected=expected):
            target = Path(path)
            if not target.exists():
                raise ValidationError("required local asset is missing")
            if expected and (not target.is_file() or file_sha(target) != expected):
                raise ValidationError("artifact checksum does not match its declared identity")
            return {"exists": True, "contents_verified": bool(expected),
                    "native_runtime_ready": None}
        check(name, verify, "Prepare or mount the selected profile's exact source, environment or artifact.")
    target = workflow.target
    benchmark = None
    if target["kind"] == "arena":
        def arena():
            native = native_configuration(workflow)
            if not Path(native.checkout).is_dir():
                raise ValidationError("native arena checkout is missing")
            if native.worker_python and not Path(native.worker_python).is_file():
                raise ValidationError("native arena worker interpreter is missing")
            return {"arena": native.arena, "benchmark_id": _ARENA_CELLS[native.arena],
                    "native_runtime_ready": None}
        check("arena", arena, "Use workflow prepare with an explicit environment/interpreter, or supply a prepared arena config.")
        def native_environment():
            return native_arena_readiness(native_configuration(workflow))
        check("arena_environment", native_environment,
              "Run workflow prepare with preparation.python and an owned environment to install native dependencies.")
    elif target["kind"] == "benchmark":
        try:
            _beir_workflow_source(workflow)
            benchmark = describe_benchmark(target["benchmark_id"])
            if (not benchmark["automatic_loader"] and not any(target.get(name) for name in
                    ("prepared_root", "beir_sample_sources", "trec_sample_sources", "longbench_sources"))):
                raise ValidationError("this benchmark requires a native adapter or separately prepared data")
            if benchmark["loader_kind"] == "checkpoint":
                raise ValidationError("checkpoint tasks require an arena workflow with their native environment")
            family_match = True
            if workflow.run.release_id != "baseline.no_op":
                method, _ = AlgorithmRegistry.default().get_release(workflow.run.release_id)
                cell = BenchmarkRegistry.default().get(target["benchmark_id"])
                # Catalog families are discovery metadata, not the full native
                # compatibility contract. Existing configured transfers (for
                # example Q-RAG with a visible BFCL tool catalog) can be valid
                # without a matching coarse family. Do not reject them here.
                family_match = method.semantic_family in cell.families
            checks.append({"check": "benchmark", "status": "ready", "details": {
                "catalog_family_match": family_match, "native_compatibility_verified": False,
                "compatibility_policy": "existing native executor validates the configured input and scorer binding",
                "automatic_loader": benchmark["automatic_loader"],
                "cold_source_download_bytes": benchmark["source_download_bytes"],
                "dataset_ready": None}})
        except Exception as error:
            checks.append({"check": "benchmark", "status": "blocked", "error": safe_error(error, workflow),
                           "action": "Use an applicable native profile/data binding; do not substitute another task or scorer."})
        if benchmark is not None and benchmark["loader_kind"] != "checkpoint":
            check("predictor_inputs", lambda: predictor_readiness(workflow, dependencies=False),
                  "Supply this task's native predictor/retriever settings and declared model capabilities; preparation will not change the protocol.")
            check("predictor_and_scorer", lambda: predictor_readiness(workflow),
                  "Correct the native predictor options and install the selected scorer's dependencies before executing.")
            check("input_and_budget", lambda: budget_and_input_readiness(workflow, None, []),
                  "Choose an applicable exact profile or explicitly configure a sufficient budget; no budget is changed automatically.")
        if (_inspect_data and benchmark is not None and benchmark["loader_kind"] != "checkpoint"
                and not any(x["status"] == "blocked" and x["check"] in {"configuration", "benchmark", "backend"}
                            for x in checks)):
            def dataset():
                adapter = (_adapters or {}).get("adapter")
                if adapter is None:
                    adapter = load_adapter(workflow, allow_network=False, read_only=True)
                if _adapters is not None:
                    _adapters["adapter"] = adapter
                examples = adapter.examples(target["split"])
                requested = target.get("case_ids")
                selected = (list(itertools.islice(examples, target["limit"])) if requested is None else
                            [item for item in examples if item.example_id in requested])
                if not selected or (requested and {x.example_id for x in selected} != set(requested)):
                    raise ValidationError("selected native cases are missing")
                compatibility = budget_and_input_readiness(workflow, adapter, selected)
                return {"dataset_ready": True, "selected_count": len(selected), "model_calls": 0,
                        "input_and_budget": compatibility}
            check("dataset", dataset,
                  "Run workflow prepare --allow-network for locked public data, or supply the exact prepared source binding.")
    if probe_endpoint and workflow.run.backend:
        try:
            probe = _probe_endpoint(workflow)
            checks.append({"check": "endpoint_models", **probe})
        except Exception as error:
            checks.append({"check": "endpoint_models", "status": "blocked", "error": safe_error(error, workflow),
                           "action": "Check endpoint reachability, TLS and model-list protocol; no generation was attempted."})
    from .preparation import selected_preparer, validate_preparer_options
    preparation = selected_preparer(workflow)
    check("method_preparation_options", lambda: validate_preparer_options(workflow),
          "Supply the missing arguments in preparation.method.options before acquiring dependencies.")
    operations = []
    if any(c["check"] == "dataset" and c["status"] == "blocked" for c in checks):
        operations.append({"operation": "dataset", "automatic": bool(benchmark and benchmark["automatic_loader"]),
                           "requires_network_if_missing": True})
    if preparation:
        operations.append({"operation": "method", **preparation, "model_calls": 0})
    if workflow.preparation.get("sources"):
        operations.append({"operation": "pinned_method_source", "model_calls": 0})
    if target["kind"] == "arena":
        operations.append({"operation": "native_environment_and_case_selection", "model_calls": 0})
    blocked_checks = {c["check"] for c in checks if c["status"] == "blocked"}
    blocked = bool(blocked_checks)
    hard = {"profile", "backend", "benchmark", "scheduling", "method_requirements", "method_preparation_options", "input_and_budget", "native_parameters", "artifact_discovery", "predictor_inputs"}
    can_prepare = not bool(blocked_checks & hard) and ("configuration" not in blocked_checks or preparation is not None)
    if not blocked:
        disposition = "ready" if (probe_endpoint or not needs_backend) else "local_ready_endpoint_unchecked"
    elif "input_and_budget" in blocked_checks or "profile" in blocked_checks:
        disposition = "not_applicable_to_requested_workflow"
    elif blocked_checks & {"configuration", "backend", "method_requirements", "method_preparation_options", "native_parameters", "artifact_discovery", "predictor_inputs"}:
        disposition = "missing_user_input"
    elif "endpoint_models" in blocked_checks:
        disposition = "resource_unavailable"
    else:
        disposition = "preparation_required" if can_prepare else "resource_unavailable"
    needs_endpoint = needs_backend and bool(workflow.run.backend)
    endpoint_ready = not needs_endpoint or any(c["check"] == "endpoint_models" and c["status"] == "ready" for c in checks)
    return {"protocol": "agentinputlab.workflow-inspection.v1", "workflow_id": requested_id,
            "checks": checks, "can_attempt": not blocked,
            "ready_for_execution": not blocked and endpoint_ready,
            "dataset_checked": _inspect_data or target["kind"] != "benchmark",
            "disposition": disposition, "can_prepare": can_prepare,
            "preparation_plan": operations, "recipe": recipe(workflow.run.release_id),
            "endpoint_probed": bool(probe_endpoint and workflow.run.backend),
            "native_runtime_ready": None, "real_acceptance_passed": None,
            "model_calls": 0, "downloads": 0, "fit_performed": False,
            "budget_per_case": to_jsonable(workflow.run.budget),
            "max_workers": workflow.max_workers}


def load_adapter(workflow, *, allow_network=False, read_only=False):
    from agentinputlab.benchmarks import load_benchmark, load_production_benchmark_adapter
    from agentinputlab.benchmarks.beir_archive import BEIR_ARCHIVE_LOCKS, BEIRArchiveAdapter
    target = workflow.target
    cell = target["benchmark_id"]
    beir_source = _beir_workflow_source(workflow)
    if beir_source.get("kind") == "beir_archive" and "source_archive" in beir_source:
        return BEIRArchiveAdapter(benchmark_id=cell, source_archive=beir_source["source_archive"],
            split=target["split"], license_acknowledged=beir_source.get("accept_license_review", False))
    if target.get("longbench_sources"):
        from agentinputlab.benchmarks.longbench import LongBenchAdapter
        from agentinputlab.longbench_configuration import sources
        return LongBenchAdapter(benchmark_id=target["benchmark_id"], **sources(target["longbench_sources"]))
    for key, module in (("beir_sample_sources", "agentinputlab.beir_sample_configuration"),
                        ("trec_sample_sources", "agentinputlab.trec_sample_configuration")):
        if target.get(key):
            import importlib
            return importlib.import_module(module).load(target[key], benchmark_id=target["benchmark_id"],
                split=target["split"], accept_license_review=target.get("accept_license_review", False))
    if target.get("prepared_root"):
        if target["benchmark_id"] == "bfcl.v3.simple_python":
            from agentinputlab.benchmarks.bfcl_native import SourceVerifiedBFCLV3ASTBridge
            from agentinputlab.benchmarks.bfcl_v3 import load_bfcl_v3_simple_adapter
            if not target.get("bfcl_source_archive"):
                raise ValidationError("BFCL prepared_root also requires bfcl_source_archive for the native scorer")
            bridge = SourceVerifiedBFCLV3ASTBridge(source_archive=target["bfcl_source_archive"],
                                                  model_id=workflow.run.backend["model_id"])
            return load_bfcl_v3_simple_adapter(target["prepared_root"], evaluation_bridge=bridge)
        return load_production_benchmark_adapter(target["benchmark_id"], target["prepared_root"])
    if read_only:
        # The regular offline loader may normalize cached raw data and create
        # cache directories. Inspection must only open already prepared inputs.
        from agentinputlab.benchmarks.assets import BenchmarkSmokeAssetRegistry
        from agentinputlab.benchmarks.cache import (
            BFCL_SIMPLE_ASSET,
            BFCL_SIMPLE_CELL,
            _source_manifest,
            benchmark_cache_dir,
        )
        from agentinputlab.benchmarks.production import production_benchmark_specs
        root = benchmark_cache_dir(beir_source.get("cache_dir"))
        if not root.is_dir():
            raise ValidationError("dataset cache is missing; workflow prepare can acquire the locked sources")
        info = describe_benchmark(target["benchmark_id"])
        if info.get("license_review_required") and not beir_source.get("accept_license_review", False):
            raise ValidationError("selected data requires accept_license_review=True")
        cell = target["benchmark_id"]
        def cached(manifest):
            path = root / "sources" / manifest.artifact_id / manifest.revision / "payload"
            if not path.is_file() or file_sha(path) != manifest.sha256:
                raise ValidationError("locked source cache is missing or changed")
            return str(path)
        if cell == BFCL_SIMPLE_CELL:
            from agentinputlab.benchmarks.bfcl_native import SourceVerifiedBFCLV3ASTBridge
            from agentinputlab.benchmarks.bfcl_v3 import load_bfcl_v3_simple_adapter
            source = target.get("bfcl_source_archive") or cached(
                BenchmarkSmokeAssetRegistry.default().get(BFCL_SIMPLE_ASSET).artifact_manifest())
            return load_bfcl_v3_simple_adapter(root / "prepared", evaluation_bridge=SourceVerifiedBFCLV3ASTBridge(
                source_archive=source, model_id=workflow.run.backend["model_id"]))
        if cell in {x.cell_id for x in production_benchmark_specs()}:
            return load_production_benchmark_adapter(cell, root / "prepared")
        if info["loader_kind"] == "longbench_v1":
            from agentinputlab.benchmarks.longbench import LongBenchAdapter
            from agentinputlab.benchmarks.longbench_sources import manifests
            return LongBenchAdapter(benchmark_id=cell, **{name: cached(value) for name, value in manifests().items()})
        if info["loader_kind"] == "beir_archive":
            return BEIRArchiveAdapter(benchmark_id=cell, source_archive=cached(_source_manifest(cell)),
                split=target["split"], license_acknowledged=True)
        raise ValidationError("selected data needs its native prepared source binding")
    runtime = {"model_id": workflow.run.backend.get("model_id")} if target["benchmark_id"] == "bfcl.v3.simple_python" else {}
    if cell in BEIR_ARCHIVE_LOCKS:
        runtime["split"] = target["split"]
    return load_benchmark(cell, cache_dir=beir_source.get("cache_dir"),
        offline=not allow_network or beir_source.get("offline", False),
        accept_license_review=beir_source.get("accept_license_review", False),
        max_download_bytes=beir_source.get("max_download_bytes", 512 * 1024 * 1024),
        max_prepared_bytes=target.get("max_prepared_bytes", 512 * 1024 * 1024),
        source_archive=target.get("bfcl_source_archive"), **runtime)


def prepare_workflow(workflow, *, output_dir, allow_network=False, environment=None, python=None):
    """Prepare exact data/case bindings and optionally an owned arena worker.

    Preparation never generates model text or fits an artifact. Source/model
    workers which need special preparation keep their existing explicit recipes.
    """
    if type(allow_network) is not bool:
        raise ValidationError("allow_network must be boolean")
    from .configuration import read_json
    from .preparation import (
        prepare_assets,
        prepare_method,
        prepare_sources,
        prepare_worker_environment,
        save,
        selected_preparer,
        validate_preparer_options,
    )
    output = Path(output_dir).absolute()
    requested_id = workflow.identity
    if output.is_symlink():
        raise ValidationError("workflow preparation output must be an owned directory")
    if output.exists():
        if not (output / "intent.json").is_file() or read_json(output / "intent.json").get("workflow_id") != requested_id:
            raise ValidationError("workflow preparation output belongs to a different workflow")
        if (output / "receipt.json").is_file():
            receipt = read_json(output / "receipt.json")
            ready = WorkflowConfig.from_file(receipt["workflow_path"])
            if file_sha(receipt["workflow_path"]) != receipt["workflow_sha256"]:
                raise ValidationError("prepared workflow changed")
            ready.verify_bindings()
            return {**receipt, "reused_preparation": True, "inspection": inspect_workflow(ready)}
    run, derived = complete_configuration(workflow.run, benchmark_id=workflow.target.get("benchmark_id"),
                                          split=workflow.target.get("split"))
    workflow = replace(workflow, run=run)
    from .artifacts import discover_artifacts
    workflow, artifact_discovery = discover_artifacts(workflow)
    initial = inspect_workflow(workflow, _inspect_data=False)
    if any(item["status"] == "blocked" and item["check"] in {
            "profile", "benchmark", "scheduling", "backend", "method_requirements", "method_preparation_options", "input_and_budget", "native_parameters", "artifact_discovery", "predictor_inputs"} for item in initial["checks"]):
        raise ValidationError("workflow prerequisites block preparation; inspect this workflow before acquiring assets")
    if (any(x["check"] == "configuration" and x["status"] == "blocked" for x in initial["checks"])
            and not selected_preparer(workflow)):
        raise ValidationError("workflow configuration needs required native inputs before preparation")
    workflow.verify_bindings()
    validate_preparer_options(workflow)
    if not output.exists():
        output.mkdir(parents=True, mode=0o700)
        save(output / "intent.json", {"workflow_id": requested_id, "model_calls": 0})
    prepare_assets(workflow, output, allow_network=allow_network)
    run = prepare_sources(workflow, output, allow_network=allow_network)
    python = python or workflow.preparation.get("python")
    environment = environment or workflow.preparation.get("environment")
    if workflow.target["kind"] != "arena":
        run = prepare_worker_environment(workflow, run, output, python=python,
                                         environment=environment, allow_network=allow_network)
    workflow = replace(workflow, run=run)
    target = dict(workflow.target)
    if target.get("arena") is not None:
        native = native_configuration(workflow)
        native_path = output / "arena-input.json"
        if not native_path.exists():
            save(native_path, native)
        target.pop("arena")
        target["arena_config"] = str(native_path)
    states = dict(workflow.case_state_hashes)
    if python and target["kind"] == "arena" and not environment:
        environment = str(output / "arena-environment")
    if environment and not python and target["kind"] == "arena":
        raise ValidationError("arena environment requires an existing Python interpreter")
    selected = []
    if target["kind"] == "benchmark":
        adapter = load_adapter(workflow, allow_network=allow_network)
        examples = adapter.examples(target["split"])
        requested = target.get("case_ids")
        if requested is None:
            selected = list(itertools.islice(examples, target["limit"]))
        else:
            wanted = set(requested)
            selected = [example for example in examples if example.example_id in wanted]
            by_id = {example.example_id: example for example in selected}
            if set(by_id) != wanted or len(selected) != len(by_id):
                raise ValidationError("requested case IDs are missing or duplicated in the native inventory")
            selected = [by_id[case_id] for case_id in requested]
        if not selected:
            raise ValidationError("native case selection is empty")
        target.pop("limit", None)
        target["case_ids"] = [example.example_id for example in selected]
        observed = {example.example_id: example.visible_state().state_hash for example in selected}
        if states and states != observed:
            raise ValidationError("prepared case states differ from existing bindings")
        states = observed
    elif environment:
        if target["kind"] != "arena":
            raise ValidationError("environment preparation applies only to arena workflows")
        from agentinputlab.arena_preparation import prepare_arena
        receipt = prepare_arena(target["arena_config"], environment=environment, python=python,
                                allow_network=allow_network, toolkit_wheel=workflow.preparation.get("toolkit_wheel"))
        target["arena_config"] = receipt["arena_config"]
    if target["kind"] == "arena":
        inventory_path = output / "inventory" / "summary.json"
        if inventory_path.is_file():
            inventory = read_json(inventory_path)
        else:
            inventory = workflow.run.evaluate_arena(target["arena_config"], output_dir=output / "inventory", inventory_only=True)
        available = inventory["available_case_ids"]
        chosen = target.get("case_ids") or available[:target["limit"]]
        if not chosen or set(chosen) - set(available):
            raise ValidationError("selected arena cases are absent from native inventory")
        target.pop("limit", None)
        target["case_ids"] = list(chosen)
    state = selected[0].visible_state(budget=workflow.run.budget) if selected else None
    if selected and target.get("benchmark_id") == "bfcl.v3.simple_python":
        from agentinputlab.bfcl_configuration import visible_state
        state = visible_state(adapter, selected[0], workflow.run.budget)[0]
    run = prepare_method(workflow, workflow.run, output, state=state)
    ready = replace(workflow, run=run, target=target, case_state_hashes=states).bind_inputs()
    report = inspect_workflow(ready, _adapters={"adapter": adapter} if target["kind"] == "benchmark" else None)
    path = output / "workflow.json"
    if not path.exists():
        save(path, ready.to_dict())
    elif WorkflowConfig.from_file(path).identity != ready.identity:
        raise ValidationError("partially prepared workflow differs")
    receipt = {"workflow_path": str(path), "workflow_sha256": file_sha(path),
            "workflow_id": ready.identity, "inspection": report, "model_calls": 0,
            "fit_performed": False, "network_acquisition_allowed": allow_network,
            "derived_configuration_fields": derived}
    receipt["artifact_discovery"] = artifact_discovery
    if not (output / "receipt.json").exists():
        save(output / "receipt.json", receipt)
    return receipt
