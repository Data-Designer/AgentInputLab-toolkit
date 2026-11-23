"""One command from preparation through existing execution and native scoring."""
from dataclasses import replace
from pathlib import Path

from agentinputlab.contracts.errors import ValidationError

from .configuration import WorkflowConfig, read_json
from .execution import run_workflow
from .planning import inspect_workflow, prepare_workflow
from .preparation import save


def launch_workflow(workflow, *, output_dir=None, execute=False, allow_network=False, resume=False):
    if type(execute) is not bool or type(allow_network) is not bool or type(resume) is not bool:
        raise ValidationError("execute, allow_network and resume must be boolean")
    if not execute:
        return {"executed": False, "inspection": inspect_workflow(workflow)}
    if not output_dir:
        raise ValidationError("workflow launch requires output_dir")
    from .recipes import complete_configuration
    run, _ = complete_configuration(workflow.run, benchmark_id=workflow.target.get("benchmark_id"),
                                     split=workflow.target.get("split"))
    resolved = replace(workflow, run=run)
    from .environment import delegate, execution_python
    # Dependency installation cannot fix an incompatible profile or missing
    # user inputs. Resolve those before installing anything.
    report = inspect_workflow(resolved, _inspect_data=False)
    if not report["can_attempt"] and not report["can_prepare"]:
        return {"executed": False, "execution_blocked": True, "inspection": report}
    python = execution_python(resolved, allow_network=allow_network)
    if python:
        return delegate(workflow, python, output_dir=output_dir, allow_network=allow_network, resume=resume)
    root = Path(output_dir).absolute()
    if root.is_symlink():
        raise ValidationError("workflow launch requires an owned directory")
    if root.exists():
        if not resume:
            raise ValidationError("workflow launch output exists; use --resume")
        if read_json(root / "launch.json").get("workflow_id") != workflow.identity:
            raise ValidationError("launch configuration changed")
    elif resume:
        raise ValidationError("no saved workflow launch to resume")
    else:
        root.mkdir(parents=True, mode=0o700)
        save(root / "launch.json", {"workflow_id": workflow.identity, "model_calls": 0})
    ready_path = root / "prepared" / "workflow.json"
    if resume and ready_path.is_file() and (root / "run").is_dir():
        ready = WorkflowConfig.from_file(ready_path)
        return run_workflow(ready, output_dir=root / "run", execute=True, resume=True)
    preparation = prepare_workflow(workflow, output_dir=root / "prepared", allow_network=allow_network)
    ready = WorkflowConfig.from_file(preparation["workflow_path"])
    result = run_workflow(ready, output_dir=root / "run", execute=True)
    return {**result, "preparation": preparation}
