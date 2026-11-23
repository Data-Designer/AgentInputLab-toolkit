"""Owned native dependency environments, separate from model execution.

Only recipes validated in clean environments belong here. A preparation receipt
is not evidence that a dataset, model, method or scorer has passed acceptance.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from agentinputlab import __version__
from agentinputlab.arena_configuration import (
    NativeArenaConfig,
    _implementation_digest,
    _read,
    _save,
    _sha,
)
from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import to_jsonable
from agentinputlab.observability.monitor import _safe_error_message
from agentinputlab.runtime_bootstrap import _materialize_source, runtime_source_plan

_SOURCE_IDS = {
    "statebench": "state_bench.common_support",
    "toolsandbox": "toolsandbox.common_support",
    "tooltalk": "tooltalk.full",
    "tau_banking": "tau3.banking_knowledge",
    "tau_retail": "tau3.banking_knowledge",
}

_SCORER_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
_SCORER_FILES = ("config.json", "tokenizer_config.json", "tokenizer.json", "vocab.txt", "model.safetensors")


def _scorer_environment(root):
    config = _read(root / "intent.json")["config"]
    return {"HF_HUB_CACHE": config.get("scorer_cache") or str(root / "scorer-cache"),
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2"}


def _install_env():
    # Installation needs no model credentials and must not inherit pip --target,
    # user-site packages or the caller's development PYTHONPATH.
    allowed = {"PATH", "HOME", "USERPROFILE", "SYSTEMROOT", "TMPDIR", "TEMP", "TMP",
               "LANG", "LC_ALL", "SSL_CERT_FILE", "SSL_CERT_DIR"}
    env = {key: value for key, value in os.environ.items() if key in allowed}
    env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PIP_CONFIG_FILE=os.devnull)
    return env


def _run(command, *, log=None, timeout=1200, environment=None):
    try:
        result = subprocess.run(command, env={**_install_env(), **(environment or {})}, capture_output=True,
                                text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValidationError("preparation command could not complete: " + _safe_error_message(exc)) from None
    if log is not None:
        # Keep bounded, redacted diagnostics; never dump environment credentials.
        with Path(log).open("x", encoding="utf-8") as stream:
            stream.write(_safe_error_message(RuntimeError(result.stdout + "\n" + result.stderr)))
    if result.returncode:
        raise ValidationError(f"preparation command exited {result.returncode}; "
                              + (f"see {log}" if log else _safe_error_message(RuntimeError(result.stderr))))
    return result.stdout


def _probe_python(python):
    try:
        return json.loads(_run([python, "-I", "-c", "import json,sys; print(json.dumps(list(sys.version_info[:3])))"],
                               timeout=30))
    except (ValueError, TypeError):
        raise ValidationError("cannot identify the requested Python interpreter") from None


def _pip_python(root):
    """Find pip or prepare an owned installer without changing the caller env."""
    if importlib.util.find_spec("pip") is not None:
        return sys.executable
    tools = Path(root).parent / (Path(root).name + ".pip-tools")
    intent = {"owner": "agentinputlab.installer.v1", "python": sys.executable,
              "python_sha256": _sha(Path(sys.executable))}
    marker = tools / "intent.json"
    if tools.exists():
        if tools.is_symlink() or not marker.is_file() or _read(marker) != intent:
            raise ValidationError("installer directory is unrelated; choose another owned preparation path")
    else:
        tools.mkdir(parents=True, mode=0o700)
        _save(marker, intent)
    python = tools / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.exists():
        _run([sys.executable, "-I", "-m", "venv", str(tools / "venv")], timeout=120)
    _run([str(python), "-I", "-m", "pip", "--version"], timeout=30)
    return str(python)


def _source(root, arena):
    return runtime_source_plan(root, method_ids=(), arena_ids=(_SOURCE_IDS[arena],))[0]


def _prepare_source(source, destination, *, arena):
    pinned = replace(_source(destination.parent, arena), checkout=source)
    if source.exists():
        check = _materialize_source(pinned, allow_network=False)
        if not check["ready"]:
            raise ValidationError(check["reason"] + "; original source left unchanged")
        _run(["git", "clone", "--quiet", "--no-hardlinks", str(source), str(destination)])
        _run(["git", "-C", str(destination), "checkout", "--detach", pinned.commit])
    else:
        result = _materialize_source(replace(pinned, checkout=destination), allow_network=True)
        if not result["ready"]:
            raise ValidationError(result["reason"])


def _verify_environment(root, digest, *, arena):
    python = root / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    native_check = "import state_bench.scoring; " if arena == "statebench" else (
        "import tool_sandbox.common.evaluation; from openai import OpenAI; "
        "client=OpenAI(api_key='readiness-no-request',base_url='http://127.0.0.1:1/v1',max_retries=0); "
        "client.close(); "
    )
    environment = None
    if arena in {"tau_banking", "tau_retail"}:
        domain = "banking_knowledge" if arena == "tau_banking" else "retail"
        environment = {"TAU2_DATA_DIR": str(root / "source" / "data"),
                       "PYTHON_DOTENV_DISABLED": "1", "LITELLM_LOCAL_MODEL_COST_MAP": "True",
                       "CUDA_VISIBLE_DEVICES": ""}
        native_check = (
            f"sys.path.insert(0, {str(root / 'source' / 'src')!r}); "
            "from tau2.runner.helpers import get_tasks, get_environment_info; "
            "import tau2.evaluator.evaluator; "
            f"tasks=get_tasks({domain!r},task_split_name='train'); "
            f"info=get_environment_info({domain!r},env_kwargs="
            + ("{'retrieval_variant':'bm25'}" if arena == "tau_banking" else "{}") + "); "
            "\nif not tasks or not info.policy: raise RuntimeError('Tau native inventory/environment unavailable')\n"
        )
    if arena == "tooltalk":
        environment = _scorer_environment(root)
        native_check = (
            f"sys.path.insert(0, {str(root / 'source' / 'src')!r}); "
            "from workers.official_arenas.providers.tooltalk import _install_sent2vec_compatibility; "
            "_install_sent2vec_compatibility(); "
            "from tooltalk.apis import ALL_APIS; from tooltalk.apis.utils import semantic_str_compare; "
            "from openai import OpenAI; "
            "client=OpenAI(api_key='readiness-no-request',base_url='http://127.0.0.1:1/v1',max_retries=0); client.close(); "
            "score=float(semantic_str_compare('scorer readiness','scorer readiness')); "
            "torch=__import__('torch'); "
            "ready=(len(ALL_APIS)==28 and abs(score-1)<1e-6 and torch.version.cuda is None); "
            "\nif not ready: raise RuntimeError('ToolTalk CPU scoring readiness failed')\n"
        )
    code = (
        "import json,sys; from agentinputlab.arena_configuration import _implementation_digest; "
        + native_check +
        "print(json.dumps({'python':sys.version.split()[0], 'implementation_sha256':_implementation_digest()}))"
    )
    observed = json.loads(_run([str(python), "-I", "-c", code], timeout=180, environment=environment))
    if observed.get("implementation_sha256") != digest:
        raise ValidationError("worker implementation differs from caller; install the same toolkit wheel")
    pip = [_pip_python(root), "-m", "pip", "--isolated", "--python", str(python)]
    _run([*pip, "check"], timeout=60)
    observed["dependencies"] = _run([*pip, "freeze", "--all"], timeout=60)
    source = root / "source"
    commit = _run(["git", "-C", str(source), "rev-parse", "HEAD"], timeout=30).strip()
    if commit != _source(root, arena).commit:
        raise ValidationError("prepared native source is not at the pinned commit")
    _run(["git", "-C", str(source), "diff", "--exit-code", "HEAD", "--"], timeout=30)
    observed["source_commit"] = commit
    if arena == "tooltalk":
        snapshot = Path(environment["HF_HUB_CACHE"]) / "models--distilbert-base-uncased" / "snapshots" / _SCORER_REVISION
        observed["scorer"] = {"revision": _SCORER_REVISION, "files": {name: _sha(snapshot / name) for name in _SCORER_FILES},
                              "runtime": "cpu", "sent2vec_path": "source-locked-bert-cls-compatibility"}
    return observed


def prepare_arena(config, *, environment, python, allow_network=False, toolkit_wheel=None):
    """Prepare an owned native worker, or verify/reuse a completed one offline.

    No existing environment is modified, no model is called and no failed or
    interrupted installation is silently repeated. Other recipes need separate
    clean-install validation before being enabled.
    """
    native = config if isinstance(config, NativeArenaConfig) else NativeArenaConfig.from_file(config)
    if native.arena not in _SOURCE_IDS:
        raise ValidationError("automatic dependency recipe currently supports statebench, toolsandbox, tooltalk and Tau; "
                              "other arenas can use an explicitly prepared worker_python")
    root = Path(environment).expanduser().absolute()
    executable = str(Path(python).expanduser().absolute())
    wheel = Path(toolkit_wheel).expanduser().absolute() if toolkit_wheel else None
    if wheel is not None and (not wheel.is_file() or not wheel.name.startswith("agentinputlab-")
                              or wheel.suffix != ".whl"):
        raise ValidationError("toolkit_wheel must be an existing agentinputlab wheel")
    digest = _implementation_digest()
    intent = {"schema_version": 1, "recipe": native.arena + "-isolated.v1",
              "config": to_jsonable(native), "python": executable,
              "implementation_sha256": digest, "toolkit_version": __version__,
              "wheel_sha256": _sha(wheel) if wheel else None}
    if root.exists() or root.is_symlink():
        if root.is_symlink() or not all((root / name).is_file() for name in ("intent.json", "receipt.json", "arena.json")):
            raise ValidationError("environment directory is incomplete or unrelated; preserve it and choose a new directory")
        if _read(root / "intent.json") != intent:
            raise ValidationError("environment was prepared for a different configuration or toolkit")
        receipt = _read(root / "receipt.json")
        observed = _verify_environment(root, digest, arena=native.arena)
        if observed != receipt.get("verification") or _sha(root / "arena.json") != receipt.get("arena_config_sha256"):
            raise ValidationError("prepared environment or arena configuration changed; not reused")
        return {**receipt, "status": "reused"}
    if not allow_network:
        raise ValidationError("new native installation requires --allow-network; no files created")
    version = _probe_python(executable)
    if native.arena == "statebench" and tuple(version[:2]) < (3, 12):
        raise ValidationError("pinned STATE requires Python >=3.12; select an installed interpreter")
    if native.arena in {"tau_banking", "tau_retail"} and not (3, 12) <= tuple(version[:2]) <= (3, 13):
        raise ValidationError("pinned Tau requires Python 3.12 or 3.13; select an installed interpreter")
    if native.arena in {"toolsandbox", "tooltalk"} and not (3, 10) <= tuple(version[:2]) <= (3, 12):
        raise ValidationError("ToolSandbox/ToolTalk dependencies require Python 3.10 through 3.12 for this recipe")
    if native.arena == "tooltalk" and sys.platform != "linux":
        raise ValidationError("automatic ToolTalk CPU preparation is currently validated on Linux; use worker_python on other platforms")
    # Verify pip's target-interpreter support before creating an owned bundle.
    installer = _pip_python(root)
    _run([installer, "-m", "pip", "--python", executable, "--version"], timeout=30)
    root.mkdir(parents=True, exist_ok=False)
    _save(root / "intent.json", intent)
    try:
        _prepare_source(Path(native.checkout), root / "source", arena=native.arena)
        _run([executable, "-I", "-m", "venv", "--without-pip", str(root / "venv")], log=root / "venv.log")
        worker = root / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        # Native OpenAI 1.17 accepts HTTPX 0.28 by metadata, but its client
        # still passes the removed `proxies` argument. Keep this transport
        # constraint local to ToolSandbox; do not change STATE's dependencies.
        constraints = ["httpx==0.27.2"] if native.arena == "toolsandbox" else []
        native_dependencies = [str(root / "source"), *constraints]
        if native.arena in {"tau_banking", "tau_retail"}:
            # Native text tasks plus local BM25. No voice, training, model
            # weights or optional experiment/plotting dependencies.
            native_dependencies = [str(root / "source") + "[knowledge]"]
        if native.arena == "tooltalk":
            # The existing source-locked sent2vec BERT-CLS adapter uses only
            # these dependencies. Avoid unused gensim/spaCy and CUDA packages.
            _run([installer, "-m", "pip", "--isolated", "--python", str(worker),
                  "install", "--disable-pip-version-check", "--no-input", "--no-cache-dir",
                  "--index-url", "https://download.pytorch.org/whl/cpu", "torch==2.11.0+cpu"],
                 log=root / "cpu-install.log")
            native_dependencies = ["numpy==1.26.4", "transformers==5.15.1", "openai==3.0.0", "tqdm>=4.66,<5"]
        _run([installer, "-m", "pip", "--isolated", "--python", str(worker),
              "install", "--disable-pip-version-check", "--no-input", "--no-cache-dir",
              "--index-url", "https://pypi.org/simple",
              str(wheel) if wheel else f"agentinputlab=={__version__}", *native_dependencies],
             log=root / "install.log")
        if native.arena == "tooltalk":
            environment = _scorer_environment(root)
            # Attempt exact local cache reuse first. Only explicitly authorized
            # preparation may fetch the five pinned scorer files when missing.
            code = (
                "from huggingface_hub import hf_hub_download\n"
                "from huggingface_hub.errors import LocalEntryNotFoundError\n"
                f"for name in {_SCORER_FILES!r}:\n"
                "    try:\n"
                f"        hf_hub_download('distilbert-base-uncased',name,revision={_SCORER_REVISION!r},local_files_only=True)\n"
                "    except LocalEntryNotFoundError:\n"
                f"        hf_hub_download('distilbert-base-uncased',name,revision={_SCORER_REVISION!r})\n"
            )
            _run([str(worker), "-I", "-c", code], log=root / "scorer-prepare.log",
                 environment={**environment, "HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"})
        observed = _verify_environment(root, digest, arena=native.arena)
        prepared = replace(native, checkout=str(root / "source"), worker_python=str(worker))
        if native.arena == "tooltalk":
            prepared = replace(prepared, scorer_cache=_scorer_environment(root)["HF_HUB_CACHE"])
        _save(root / "arena.json", prepared)
        receipt = {"status": "installed", "recipe": intent["recipe"], "model_calls": 0,
                   "arena_config": str(root / "arena.json"),
                   "arena_config_sha256": _sha(root / "arena.json"), "verification": observed}
        _save(root / "receipt.json", receipt)
        return receipt
    except Exception as exc:
        _save(root / "failure.json", {"status": "failed", "error": _safe_error_message(exc), "model_calls": 0})
        raise
