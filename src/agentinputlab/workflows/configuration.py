"""One workflow file; no shell recipes, implicit fitting, or generated defaults."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from agentinputlab.configuration import RunConfig, _json_mapping, _known, _unique
from agentinputlab.contracts import FrozenAgentState
from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import content_hash, to_jsonable
from agentinputlab.experiments.secrets import require_secret_free

PROTOCOL = "agentinputlab.workflow.v1"
MAX_FILE_BYTES = 1024 * 1024
MAX_CASES = 2000
KINDS = {"query", "benchmark", "arena"}
_TARGET_FIELDS = {
    "kind", "queries", "benchmark_id", "split", "case_ids", "limit", "arena_config",
    "cache_dir", "prepared_root", "accept_license_review", "max_download_bytes",
    "max_prepared_bytes", "predictor_parameters", "bfcl_source_archive",
    "qrag_reader", "longbench_sources", "beir_sample_sources", "trec_sample_sources",
    "arena",
}
_PATHS = {"arena_config", "cache_dir", "prepared_root", "bfcl_source_archive"}
ARENA_IDS = {"tooltalk.full": "tooltalk", "state_bench.common_support": "statebench",
             "toolsandbox.common_support": "toolsandbox", "tau3.retail": "tau_retail",
             "tau3.banking_knowledge": "tau_banking"}


def native_configuration(workflow):
    from agentinputlab.arena_configuration import NativeArenaConfig
    if workflow.target.get("arena") is not None:
        return NativeArenaConfig(**to_jsonable(workflow.target["arena"]))
    return NativeArenaConfig.from_file(workflow.target["arena_config"])


def read_json(path: str | Path) -> dict[str, Any]:
    try:
        with Path(path).open("rb") as stream:
            raw = stream.read(MAX_FILE_BYTES + 1)
        if len(raw) > MAX_FILE_BYTES:
            raise ValidationError("workflow input exceeds the 1 MiB limit")
        result = json.loads(raw, object_pairs_hook=_unique)
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError):
        raise ValidationError("workflow input must be an accessible UTF-8 JSON object") from None
    return to_jsonable(_json_mapping(result, "workflow input"))


def file_sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class WorkflowConfig:
    """Run settings and a finite task selection, independent of output location.

    Concurrency is an execution setting, never an increase to per-case budgets.
    The workflow layer neither fits artifacts nor chooses a different profile.
    """

    run: RunConfig
    target: Mapping[str, Any]
    max_workers: int = 1
    required_capabilities: tuple[str, ...] = ()
    input_bindings: Mapping[str, str] = field(default_factory=dict)
    case_state_hashes: Mapping[str, str] = field(default_factory=dict)
    preparation: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.run, RunConfig):
            raise ValidationError("workflow.run must be a RunConfig")
        object.__setattr__(self, "run", self.run.resolve_paths(Path.cwd()))
        target = dict(_json_mapping(self.target, "workflow target"))
        _known(target, _TARGET_FIELDS, "workflow target")
        kind = target.get("kind")
        if not isinstance(kind, str) or kind not in KINDS:
            raise ValidationError("workflow target.kind must be query, benchmark, or arena")
        if type(self.max_workers) is not int or not 1 <= self.max_workers <= 4:
            raise ValidationError("workflow max_workers must be between 1 and 4")
        if not isinstance(self.required_capabilities, (list, tuple)):
            raise ValidationError("required_capabilities must be an array")
        capabilities = tuple(self.required_capabilities)
        if (any(not isinstance(x, str) or not x for x in capabilities)
                or len(set(capabilities)) != len(capabilities)):
            raise ValidationError("required_capabilities must be unique capability names")
        from agentinputlab.backends.llm import ModelCapability
        try:
            tuple(ModelCapability(x) for x in capabilities)
        except ValueError:
            raise ValidationError("unknown required model capability") from None
        if kind == "query":
            _known(target, {"kind", "queries"}, "query target")
            queries = target.get("queries")
            if (not isinstance(queries, (tuple, list)) or not 1 <= len(queries) <= MAX_CASES
                    or any(not isinstance(x, str) or not x.strip() for x in queries)):
                raise ValidationError("query workflow requires 1..2000 non-empty queries")
            # Use the existing state validation; no model or constructor is run.
            for query in queries:
                FrozenAgentState(query=query)
        else:
            cases = target.get("case_ids")
            limit = target.get("limit")
            if ((cases is None) == (limit is None)
                    or (limit is not None and (type(limit) is not int or not 1 <= limit <= MAX_CASES))):
                raise ValidationError("choose case_ids or a limit in [1, 2000]")
            if cases is not None and (not isinstance(cases, (tuple, list)) or not 1 <= len(cases) <= MAX_CASES
                    or any(not isinstance(x, str) or not x.strip() for x in cases)
                    or len(set(cases)) != len(cases)):
                raise ValidationError("case_ids must be 1..2000 unique non-empty strings")
            if kind == "arena":
                _known(target, {"kind", "case_ids", "limit", "arena_config", "arena"}, "arena target")
                if (target.get("arena") is None) == (target.get("arena_config") is None):
                    raise ValidationError("arena workflow requires arena settings or an arena_config path")
                if "arena" in target:
                    _json_mapping(target["arena"], "native arena settings")
                elif not isinstance(target["arena_config"], str) or not target["arena_config"].strip():
                    raise ValidationError("arena_config must be a nonempty local path")
            else:
                if "arena_config" in target or "queries" in target:
                    raise ValidationError("benchmark target cannot contain arena_config or queries")
                if not isinstance(target.get("benchmark_id"), str) or not target["benchmark_id"].strip():
                    raise ValidationError("benchmark workflow requires benchmark_id")
                if not isinstance(target.get("split"), str) or not target["split"].strip():
                    raise ValidationError("benchmark workflow requires an explicit split")
                if target.get("prepared_root") and target.get("cache_dir"):
                    raise ValidationError("choose prepared_root or cache_dir")
                if type(target.get("accept_license_review", False)) is not bool:
                    raise ValidationError("accept_license_review must be boolean")
                for name in ("max_download_bytes", "max_prepared_bytes"):
                    if name in target and (type(target[name]) is not int or target[name] < 1):
                        raise ValidationError(f"target.{name} must be a positive integer")
                if "predictor_parameters" in target:
                    _json_mapping(target["predictor_parameters"], "predictor_parameters")
                for name in ("qrag_reader", "longbench_sources", "beir_sample_sources", "trec_sample_sources"):
                    if name in target:
                        _json_mapping(target[name], name)
        for name in _PATHS & target.keys():
            if not isinstance(target[name], str) or not target[name].strip() or "\x00" in target[name]:
                raise ValidationError(f"target.{name} must be an explicit local path")
            target[name] = str(Path(target[name]).expanduser().absolute())
        references = _json_mapping(self.input_bindings, "input bindings")
        bindings = {str(Path(k).expanduser().absolute()): v for k, v in references.items()}
        if any(not isinstance(k, str) or not isinstance(v, str) or len(v) != 64
               or any(c not in "0123456789abcdef" for c in v) for k, v in bindings.items()):
            raise ValidationError("input_bindings must map paths to SHA256 values")
        object.__setattr__(self, "target", _json_mapping(target, "target"))
        object.__setattr__(self, "required_capabilities", capabilities)
        object.__setattr__(self, "input_bindings", _json_mapping(bindings, "input bindings"))
        states = dict(self.case_state_hashes)
        if any(not isinstance(k, str) or not k or not isinstance(v, str) or len(v) != 64
               or any(c not in "0123456789abcdef" for c in v) for k, v in states.items()):
            raise ValidationError("case_state_hashes must map case IDs to SHA256 values")
        object.__setattr__(self, "case_state_hashes", _json_mapping(states, "case states"))
        preparation = dict(_json_mapping(self.preparation, "preparation"))
        _known(preparation, {"python", "environment", "sources", "source_root", "method", "assets", "max_download_bytes", "toolkit_wheel", "artifact_catalog"}, "preparation")
        for name in ("python", "environment", "source_root", "toolkit_wheel", "artifact_catalog"):
            if name in preparation:
                path = preparation[name]
                if not isinstance(path, str) or not path.strip() or "\x00" in path:
                    raise ValidationError(f"preparation.{name} must be a local path")
                preparation[name] = str(Path(path).expanduser().absolute())
        if "sources" in preparation and type(preparation["sources"]) is not bool:
            raise ValidationError("preparation.sources must be boolean")
        limit = preparation.get("max_download_bytes", 512 * 1024 * 1024)
        if type(limit) is not int or limit < 1:
            raise ValidationError("preparation.max_download_bytes must be positive")
        if "assets" in preparation:
            assets = preparation["assets"]
            if not isinstance(assets, (tuple, list)) or len(assets) > 1000:
                raise ValidationError("preparation.assets must be a bounded array")
            from .preparation import validate_asset
            preparation["assets"] = [validate_asset(item) for item in assets]
        if "method" in preparation:
            from .recipes import PREPARERS
            method = dict(_json_mapping(preparation["method"], "method preparation"))
            _known(method, {"kind", "options"}, "method preparation")
            if method.get("kind") not in PREPARERS:
                raise ValidationError("unknown method preparation recipe")
            method["options"] = dict(_json_mapping(method.get("options", {}), "preparation options"))
            preparation["method"] = method
        object.__setattr__(self, "preparation", _json_mapping(preparation, "preparation"))
        require_secret_free(self.to_dict(), label="workflow")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, base_dir: str | Path | None = None):
        value = dict(_json_mapping(value, "workflow"))
        _known(value, {"schema_version", "protocol", "run", "target", "max_workers",
                       "required_capabilities", "input_bindings", "case_state_hashes", "preparation"}, "workflow")
        version = value.pop("schema_version", None)
        if type(version) is not int or version != 1 or value.get("protocol") != PROTOCOL:
            raise ValidationError("unsupported workflow schema or protocol")
        value.pop("protocol")
        if "run" not in value or "target" not in value:
            raise ValidationError("workflow requires run and target objects")
        run = RunConfig.from_dict(value.pop("run"))
        target = dict(_json_mapping(value.pop("target"), "target"))
        if base_dir is not None:
            base = Path(base_dir).absolute()
            run = run.resolve_paths(base)
            for name in _PATHS & target.keys():
                path = target[name]
                if isinstance(path, str) and path and not Path(path).expanduser().is_absolute():
                    target[name] = str(base / path)
            def source_paths(value):
                if isinstance(value, Mapping):
                    return {key: (str(base / Path(item).expanduser()) if isinstance(item, str)
                            and (key == "path" or key.endswith(("_path", "_root")) or key in
                                 {"data_zip", "scorer_path", "prompts_path", "checkout", "worker_python", "scorer_cache"})
                            else source_paths(item)) for key, item in value.items()}
                if isinstance(value, (list, tuple)):
                    return [source_paths(item) for item in value]
                return value
            for name in ("qrag_reader", "longbench_sources", "beir_sample_sources", "trec_sample_sources", "predictor_parameters", "arena"):
                if name in target:
                    target[name] = source_paths(target[name])
            if "input_bindings" in value:
                value["input_bindings"] = {str(Path(k) if Path(k).is_absolute() else base / k): v
                                           for k, v in dict(value["input_bindings"]).items()}
            if "preparation" in value:
                preparation = dict(value["preparation"])
                for name in ("python", "environment", "source_root", "toolkit_wheel", "artifact_catalog"):
                    if isinstance(preparation.get(name), str):
                        preparation[name] = str(base / Path(preparation[name]).expanduser())
                if "assets" in preparation:
                    preparation["assets"] = [{**asset, "destination": str(base / Path(asset["destination"]).expanduser())}
                                              for asset in preparation["assets"]]
                value["preparation"] = preparation
        return cls(run=run, target=target, **value)

    @classmethod
    def from_file(cls, path: str | Path):
        return cls.from_dict(read_json(path), base_dir=Path(path).absolute().parent)

    def to_dict(self):
        return {"schema_version": 1, "protocol": PROTOCOL, "run": self.run.to_dict(),
                "target": to_jsonable(self.target), "max_workers": self.max_workers,
                "required_capabilities": list(self.required_capabilities),
                "input_bindings": dict(self.input_bindings),
            "case_state_hashes": dict(self.case_state_hashes), "preparation": to_jsonable(self.preparation)}

    @property
    def identity(self):
        value = self.to_dict()
        value.pop("max_workers")  # Safe to change scheduling on the same frozen work.
        return content_hash(value)

    def verify_bindings(self):
        for path, expected in self.input_bindings.items():
            try:
                if file_sha(path) != expected:
                    raise ValidationError("a bound workflow input has changed; create a new workflow")
            except OSError:
                raise ValidationError("a bound workflow input is missing") from None

    def bind_inputs(self):
        """Bind small configuration/worker/artifact files, never whole model trees."""
        paths = set()
        if self.target.get("arena_config"):
            paths.add(self.target["arena_config"])
        if self.preparation.get("toolkit_wheel"):
            paths.add(self.preparation["toolkit_wheel"])
        if self.preparation.get("artifact_catalog"):
            paths.add(self.preparation["artifact_catalog"])
        for reference in self.run.artifacts.values():
            if isinstance(reference, Mapping) and reference.get("path"):
                paths.add(reference["path"])
        for name, value in self.run.runtime.items():
            if isinstance(value, str) and (name.endswith("_path") or name in {
                    "python_executable", "tokenizer_manifest", "compatibility_manifest_path"}):
                paths.add(value)
        bindings = dict(self.input_bindings)
        for path in paths:
            source = Path(path).absolute()
            if source.is_file():
                if source.stat().st_size > 64 * 1024 * 1024:
                    raise ValidationError("workflow binding file exceeds 64 MiB; use a manifest-bound native profile")
                actual = file_sha(source)
                if str(source) in bindings and bindings[str(source)] != actual:
                    raise ValidationError("a previously bound input has changed")
                bindings[str(source)] = actual
        return replace(self, input_bindings=bindings)


def workflow_template(*, release_id="plan.plan_and_solve.ps_prompt", query="Find a suitable flight",
                      run: RunConfig | None = None, benchmark_id: str | None = None,
                      case_ids: tuple[str, ...] | None = None, limit: int = 1,
                      cache_dir: str | None = None, arena_config: str | None = None,
                      backend=None, parameters=None, budget=None, retriever=None,
                      artifacts=None, runtime=None, preparation=None, environment_backend=None):
    """Generate an editable draft, preserving supplied native parameters.

    Required method inputs are never invented. Inspect the draft for actionable
    requirements before preparation or execution.
    """
    if arena_config and benchmark_id:
        raise ValidationError("choose a benchmark or an arena configuration")
    if run is not None and any(x is not None for x in (backend, parameters, budget, retriever, artifacts, runtime)):
        raise ValidationError("choose a complete run configuration or individual configuration fields")
    options = {k: v for k, v in dict(backend=backend, parameters=parameters, budget=budget,
                                   retriever=retriever, artifacts=artifacts, runtime=runtime).items() if v is not None}
    supplied_run = run is not None
    run = run or RunConfig(release_id, **options)
    if arena_config:
        target = {"kind": "arena", "arena_config": arena_config}
    elif benchmark_id:
        from agentinputlab.benchmarks import describe_benchmark
        info = describe_benchmark(benchmark_id)
        if benchmark_id in ARENA_IDS:
            from agentinputlab.benchmarks.cache import benchmark_cache_dir
            source = benchmark_cache_dir(cache_dir) / "runtime-sources" / "_arenas" / benchmark_id / "checkout"
            target = {"kind": "arena", "arena": {"arena": ARENA_IDS[benchmark_id], "checkout": str(source)}}
            if environment_backend is not None:
                target["arena"]["environment_backend"] = environment_backend
        else:
            target = {"kind": "benchmark", "benchmark_id": benchmark_id,
                      "split": info["split"] or "test"}
            if cache_dir:
                target["cache_dir"] = cache_dir
    else:
        return WorkflowConfig(run, {"kind": "query", "queries": [query]}, preparation=preparation or {})
    from .recipes import complete_configuration
    run, _ = complete_configuration(run, benchmark_id=benchmark_id, split=target.get("split"))
    preparation = dict(preparation or {})
    from .recipes import WORKER_KINDS, recipe
    contract = recipe(run.release_id)
    if not supplied_run and not run.runtime and len(contract["runtime_kinds"]) == 1:
        kind = contract["runtime_kinds"][0]
        if kind in WORKER_KINDS:
            import sys

            from agentinputlab.benchmarks.cache import benchmark_cache_dir
            from agentinputlab.registry import AlgorithmRegistry
            method, _ = AlgorithmRegistry.default().get_release(run.release_id)
            root = Path(preparation.get("source_root", benchmark_cache_dir(cache_dir) / "runtime-sources"))
            run = replace(run, runtime={"kind": kind, "checkout": str(root / method.method_id / "checkout"),
                                       "python_executable": preparation.get("python", sys.executable)})
            preparation.setdefault("sources", True)
            preparation.setdefault("source_root", str(root))
    if (not supplied_run and not run.retriever and contract["retriever_required"]
            and benchmark_id and not benchmark_id.startswith(("beir.", "bright.", "trec_dl."))):
        run = replace(run, retriever={"kind": "visible_context_bm25"})
    if run.release_id == "plan.plan_and_budget.full" and not run.runtime:
        from agentinputlab.benchmarks.cache import benchmark_cache_dir
        root = Path(preparation.get("source_root", benchmark_cache_dir(cache_dir) / "runtime-sources"))
        run = replace(run, runtime={"kind": "plan_and_budget_native", "source_root": str(root / "plan_and_budget" / "checkout")})
        preparation.setdefault("sources", True)
        preparation.setdefault("source_root", str(root))
    target.update({"case_ids": list(case_ids)} if case_ids else {"limit": limit})
    return WorkflowConfig(run, target, preparation=preparation)
