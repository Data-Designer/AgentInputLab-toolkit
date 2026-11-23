"""CLI wiring for existing configuration/preparation/native evaluation paths."""
import json
from pathlib import Path

from agentinputlab.configuration import RunConfig
from agentinputlab.contracts.errors import ValidationError

from .configuration import WorkflowConfig, workflow_template
from .execution import run_workflow, workflow_status
from .launch import launch_workflow
from .planning import inspect_workflow, prepare_workflow
from .recipes import list_recipes


def _output(value):
    print(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def _command(args):
    action = args.workflow_action
    if action == "recipes":
        _output(list_recipes())
        return 0
    if action == "init":
        if args.run_config and args.release_id:
            raise ValidationError("choose --run-config or --release-id")
        run = RunConfig.from_file(args.run_config) if args.run_config else None
        backend = None
        if args.model_id or args.base_url or args.deployment_revision:
            if run is not None or not all((args.model_id, args.base_url, args.deployment_revision)):
                raise ValidationError("supply model-id, base-url and deployment-revision together, or reuse run-config")
            backend = {"model_id": args.model_id, "base_url": args.base_url,
                       "deployment_identity": {"revision": args.deployment_revision}, "api_key_env": args.api_key_env}
        from .configuration import read_json
        fields = {name: read_json(path) for name in ("parameters", "retriever", "artifacts", "runtime", "preparation")
                  if (path := getattr(args, name + "_config", None))}
        if args.budget_config:
            from agentinputlab import ExecutionBudget
            fields["budget"] = ExecutionBudget(**read_json(args.budget_config))
        if args.python:
            fields["preparation"] = {**fields.get("preparation", {}), "python": args.python}
        if args.artifact_catalog:
            fields["preparation"] = {**fields.get("preparation", {}), "artifact_catalog": args.artifact_catalog}
        if args.environment_backend_config:
            fields["environment_backend"] = read_json(args.environment_backend_config)
        workflow = workflow_template(run=run, backend=backend, **fields,
            release_id=args.release_id or "plan.plan_and_solve.ps_prompt", query=args.query,
            benchmark_id=args.benchmark, case_ids=tuple(args.case_id) if args.case_id else None,
            limit=args.limit, cache_dir=args.cache_dir, arena_config=args.arena_config)
        value = workflow.to_dict()
        for option, name in (("predictor_config", "predictor_parameters"), ("longbench_config", "longbench_sources"),
                             ("beir_sample_config", "beir_sample_sources"), ("trec_sample_config", "trec_sample_sources")):
            path = getattr(args, option)
            if path:
                value["target"][name] = read_json(path)
                workflow = WorkflowConfig.from_dict(value, base_dir=Path(path).absolute().parent)
                value = workflow.to_dict()
        for name in ("prepared_root", "bfcl_source_archive"):
            if getattr(args, name):
                value["target"][name] = str(Path(getattr(args, name)).expanduser().absolute())
        if args.accept_license_review:
            value["target"]["accept_license_review"] = True
        workflow = WorkflowConfig.from_dict(value)
        _output(workflow.to_dict())
        return 0
    if action == "status":
        value = workflow_status(args.output_dir)
    else:
        workflow = WorkflowConfig.from_file(args.path)
        if action in {"inspect", "plan"}:
            value = inspect_workflow(workflow, probe_endpoint=args.probe_endpoint)
            _output(value)
            return 0 if value["can_attempt"] else 2
        if action == "prepare":
            value = prepare_workflow(workflow, output_dir=args.output_dir,
                allow_network=args.allow_network, environment=args.environment, python=args.python)
            _output(value)
            return 0 if value["inspection"]["can_attempt"] else 2
        if action == "launch":
            value = launch_workflow(workflow, output_dir=args.output_dir, execute=args.execute,
                                    allow_network=args.allow_network, resume=args.resume)
        else:
            value = run_workflow(workflow, output_dir=args.output_dir, execute=args.execute, resume=args.resume)
    _output(value)
    if (value.get("execution_blocked") or value.get("status") == "needs_reconciliation"
            or ("inspection" in value and not value["inspection"]["can_attempt"])):
        return 2
    if value.get("status") in {"completed_with_failures", "awaiting_user"}:
        return 1
    return 0


def add_parser(subparsers):
    parser = subparsers.add_parser("workflow", help="Connect configuration, preparation, execution and native scoring.")
    commands = parser.add_subparsers(dest="workflow_action", required=True)
    recipes = commands.add_parser("recipes", help="List exact profile contracts and preparation routes; no model calls.")
    recipes.set_defaults(func=_command)
    init = commands.add_parser("init", help="Print an editable workflow; never invent native required inputs.")
    init.add_argument("--run-config", type=Path, help="Reuse a complete existing run configuration.")
    init.add_argument("--release-id", help="Exact profile ID; defaults to the prompt-only Plan-and-Solve starter.")
    init.add_argument("--query", default="Find a suitable flight")
    init.add_argument("--model-id", help="Exact model ID served by the provider.")
    init.add_argument("--base-url", help="OpenAI-compatible API root, usually ending in /v1.")
    init.add_argument("--deployment-revision", help="Provider/model deployment revision.")
    init.add_argument("--api-key-env", default="OPENAI_API_KEY")
    init.add_argument("--python", help="Installed interpreter for automatic owned native environment preparation.")
    init.add_argument("--artifact-catalog", help="Exact-model/benchmark JSON bindings for reusing existing fitted artifacts; never fits on test data.")
    init.add_argument("--environment-backend-config", help="Explicit simulator/judge model settings for arenas that require them.")
    for name in ("parameters", "retriever", "artifacts", "runtime", "preparation"):
        init.add_argument("--" + name + "-config", help="JSON " + name + " settings; existing native field names are preserved.")
    init.add_argument("--budget-config", help="ExecutionBudget JSON; explicit call/token/time limits are preserved.")
    init.add_argument("--predictor-config", help="Existing native predictor JSON, including explicit paid reader protocols.")
    init.add_argument("--prepared-root", help="Reuse native normalized data, without downloading another copy.")
    init.add_argument("--bfcl-source-archive", help="Pinned source archive for the native BFCL scorer.")
    init.add_argument("--longbench-config", help="Existing LongBench source configuration JSON.")
    init.add_argument("--beir-sample-config", help="Existing bounded BEIR source configuration JSON.")
    init.add_argument("--trec-sample-config", help="Existing bounded TREC DL source configuration JSON.")
    init.add_argument("--accept-license-review", action="store_true", help="Confirm you reviewed the selected dataset terms.")
    target = init.add_mutually_exclusive_group()
    target.add_argument("--benchmark")
    target.add_argument("--arena-config")
    selection = init.add_mutually_exclusive_group()
    selection.add_argument("--case-id", action="append")
    selection.add_argument("--limit", type=int, default=1)
    init.add_argument("--cache-dir")
    init.set_defaults(func=_command)
    plan = commands.add_parser("inspect", aliases=["plan"], help="Summarize prerequisites; zero downloads or generation.")
    plan.add_argument("path")
    plan.add_argument("--probe-endpoint", action="store_true", help="Make one GET /models to the configured provider; never generate.")
    plan.set_defaults(func=_command)
    prepare = commands.add_parser("prepare", help="Reuse existing bounded data/arena preparation and freeze case selection.")
    prepare.add_argument("path")
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--allow-network", action="store_true", help="Allow the existing pinned asset loaders to acquire missing files.")
    prepare.add_argument("--environment", help="Owned arena environment path, passed to the existing prepare-arena recipe.")
    prepare.add_argument("--python", help="Existing interpreter for arena preparation; no Python download.")
    prepare.set_defaults(func=_command)
    run = commands.add_parser("run", help="Inspect by default; --execute runs selected cases once through existing APIs.")
    run.add_argument("path")
    run.add_argument("--output-dir")
    run.add_argument("--execute", action="store_true")
    run.add_argument("--resume", action="store_true", help="Skip all started attempts, including failed and uncertain ones.")
    run.set_defaults(func=_command)
    launch = commands.add_parser("launch", help="Prepare missing configured dependencies, execute and score; dry by default.")
    launch.add_argument("path")
    launch.add_argument("--output-dir")
    launch.add_argument("--execute", action="store_true")
    launch.add_argument("--resume", action="store_true")
    launch.add_argument("--allow-network", action="store_true")
    launch.set_defaults(func=_command)
    status = commands.add_parser("status", help="Read verified journal status without rerunning anything.")
    status.add_argument("output_dir")
    status.set_defaults(func=_command)
