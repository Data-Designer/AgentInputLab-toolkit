"""Select an existing runtime or install small execution extras in an owned env."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import content_hash

from .configuration import file_sha, read_json
from .preparation import save
from .readiness import interpreter_readiness


def requirements(workflow):
    """Only existing, bounded CPU extras are installed automatically here."""
    extras, modules = [], []
    if workflow.run.runtime.get("kind") == "plan_and_budget_native":
        extras.append("plan-and-budget")
        modules.append("numpy")
    benchmark = workflow.target.get("benchmark_id", "")
    if benchmark.startswith(("beir.", "bright.", "trec_dl.")):
        extras.append("evaluation")
        modules.append("ir_measures")
    if benchmark.startswith("longbench."):
        extras.append("longbench")
        modules.extend(("rouge", "jieba", "fuzzywuzzy"))
    if benchmark == "hotpotqa.distractor" and not workflow.target.get("prepared_root"):
        from agentinputlab.benchmarks.assets import BenchmarkSmokeAssetRegistry
        from agentinputlab.benchmarks.cache import benchmark_cache_dir
        from agentinputlab.benchmarks.preprocess import _prepared_path
        asset = BenchmarkSmokeAssetRegistry.default().get("hotpotqa.distractor.validation")
        if not _prepared_path(asset, benchmark_cache_dir(workflow.target.get("cache_dir")) / "prepared").exists():
            extras.append("benchmarks")
            modules.append("pyarrow")
    return sorted(set(extras)), sorted(set(modules))


def execution_python(workflow, *, allow_network):
    extras, modules = requirements(workflow)
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    if not missing:
        return None
    chosen = workflow.preparation.get("python")
    if chosen and Path(chosen).absolute() != Path(sys.executable).absolute():
        try:
            interpreter_readiness(chosen, modules)
            return str(chosen)
        except ValidationError:
            pass
    from agentinputlab import __version__
    from agentinputlab.arena_preparation import _pip_python, _run
    from agentinputlab.benchmarks.cache import benchmark_cache_dir
    wheel = workflow.preparation.get("toolkit_wheel")
    intent = {"owner": "agentinputlab.execution-environment.v1", "python": chosen or sys.executable,
              "extras": extras, "toolkit_version": __version__, "wheel_sha256": file_sha(wheel) if wheel else None}
    root = benchmark_cache_dir(workflow.target.get("cache_dir")) / "execution-environments" / content_hash(intent)[:24]
    python = root / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if root.exists():
        if root.is_symlink() or not (root / "intent.json").is_file() or read_json(root / "intent.json") != intent:
            raise ValidationError("execution environment belongs to another configuration")
        if (root / "receipt.json").is_file():
            interpreter_readiness(python, modules)
            return str(python)
    else:
        if not allow_network:
            raise ValidationError("missing execution dependencies " + ", ".join(missing) + "; use launch --allow-network or preparation.python")
        root.mkdir(parents=True, mode=0o700)
        save(root / "intent.json", intent)
    if not allow_network:
        raise ValidationError("incomplete execution environment needs --allow-network")
    if not python.exists():
        _run([chosen or sys.executable, "-I", "-m", "venv", "--without-pip", str(root / "venv")])
    package = "agentinputlab[" + ",".join(extras) + "]"
    package += " @ " + Path(wheel).as_uri() if wheel else "==" + __version__
    _run([_pip_python(root), "-m", "pip", "--isolated", "--python", str(python), "install",
          "--no-input", "--no-cache-dir", "--disable-pip-version-check", package])
    observed = interpreter_readiness(python, modules)
    if not (root / "receipt.json").exists():
        save(root / "receipt.json", {"verification": observed, "model_calls": 0, "fit_performed": False})
    return str(python)


def _package_bootstrap():
    """Bind our code without leaking the caller's binary site-packages."""
    from agentinputlab import __file__ as package_file
    from workers.official_arenas import protocol
    core = str(Path(package_file).parent)
    workers_root = str(Path(protocol.__file__).parents[1])
    # PathFinder creates a real namespace path/resource reader. A manually
    # assigned list works for imports but breaks importlib.resources.files().
    return ("import importlib.util,importlib.machinery,sys\n"
            f"spec=importlib.machinery.PathFinder.find_spec('workers', {[str(Path(workers_root).parent)]!r})\n"
            "module=importlib.util.module_from_spec(spec);sys.modules['workers']=module\n"
            f"spec=importlib.util.spec_from_file_location('agentinputlab', {str(Path(core) / '__init__.py')!r}, "
            f"submodule_search_locations={[core]!r})\n"
            "module=importlib.util.module_from_spec(spec);sys.modules['agentinputlab']=module;spec.loader.exec_module(module)\n")


def _child_environment(workflow):
    """Keep native descendants on this pure toolkit code, not an older install.

    Only an owned bootstrap directory enters PYTHONPATH; the caller's entire
    site-packages (and its binary extensions) never enter the selected runtime.
    Explicit native worker interpreters keep their independent digest gate.
    """
    from agentinputlab.arena_configuration import _implementation_digest
    from agentinputlab.benchmarks.cache import benchmark_cache_dir
    code = _package_bootstrap()
    identity = {"implementation": _implementation_digest(), "bootstrap": code}
    root = benchmark_cache_dir(workflow.target.get("cache_dir")) / "workflow-code" / content_hash(identity)[:24]
    path = root / "sitecustomize.py"
    if root.exists():
        if root.is_symlink() or path.is_symlink() or not path.is_file() or path.read_text() != code:
            raise ValidationError("delegated code bootstrap changed; no child was launched")
    else:
        root.mkdir(parents=True, mode=0o700)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(code)
            stream.flush()
            os.fsync(stream.fileno())
    return {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(root), "PYTHONNOUSERSITE": "1"}


def delegate(workflow, python, *, output_dir, allow_network, resume):
    """Run the same installed toolkit code with the selected dependencies."""
    from agentinputlab.benchmarks.cache import benchmark_cache_dir
    cache = benchmark_cache_dir(workflow.target.get("cache_dir")) / "workflow-inputs"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / (workflow.identity + ".json")
    if not path.exists():
        save(path, workflow.to_dict())
    elif content_hash(read_json(path)) != content_hash(workflow.to_dict()):
        raise ValidationError("delegated workflow input changed")
    # Bind only our two packages. Adding the caller's entire site-packages
    # would leak incompatible binary dependencies into the chosen Python.
    code = _package_bootstrap()
    code += "from agentinputlab.cli.main import main; raise SystemExit(main(sys.argv[1:]))"
    args = [python, "-I", "-c", code, "workflow", "launch", str(path), "--execute", "--output-dir", str(output_dir)]
    if allow_network:
        args.append("--allow-network")
    if resume:
        args.append("--resume")
    env = _child_environment(workflow)
    # This wait does not retry, time out or terminate a model-consuming child.
    completed = subprocess.run(args, env=env, capture_output=True, text=True, check=False)
    try:
        result = json.loads(completed.stdout)
    except (ValueError, TypeError):
        raise ValidationError("execution environment failed; no model attempt was automatically repeated") from None
    return {**result, "execution_python": python}
