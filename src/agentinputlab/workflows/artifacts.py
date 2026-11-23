"""Resolve caller-owned, exact-model artifact bindings without fitting anything."""
from dataclasses import replace
from pathlib import Path

from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import content_hash, to_jsonable

from .configuration import ARENA_IDS, native_configuration, read_json


def binding_identity(workflow):
    """Addresses are execution locations, not permissions to change a model."""
    backend = workflow.run.backend
    if not backend.get("model_id") or not backend.get("deployment_identity"):
        raise ValidationError("artifact discovery requires model_id and an explicit deployment_identity")
    benchmark = workflow.target.get("benchmark_id")
    if workflow.target["kind"] == "arena":
        arena = native_configuration(workflow).arena
        benchmark = next(key for key, value in ARENA_IDS.items() if value == arena)
    if not benchmark:
        raise ValidationError("artifact discovery requires a benchmark or native arena, not an unbound query")
    return {"release_id": workflow.run.release_id, "benchmark_id": benchmark,
            "model_id": backend["model_id"], "deployment_identity": to_jsonable(backend["deployment_identity"])}


def discover_artifacts(workflow):
    """Explicit run.artifacts wins; never search arbitrary experiment folders."""
    source = workflow.preparation.get("artifact_catalog")
    if not source or workflow.run.artifacts:
        return workflow, {"selected": False, "explicit_bindings_preserved": bool(workflow.run.artifacts)}
    identity = binding_identity(workflow)
    payload = read_json(source)
    if (set(payload) != {"schema_version", "bindings"} or type(payload["schema_version"]) is not int
            or payload["schema_version"] != 1):
        raise ValidationError("artifact catalog requires schema_version=1 and bindings")
    rows = payload["bindings"]
    if not isinstance(rows, list) or len(rows) > 1000:
        raise ValidationError("artifact catalog bindings must be a bounded array")
    matches = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"identity", "artifacts"}:
            raise ValidationError("artifact catalog entries require identity and artifacts")
        if not isinstance(row["identity"], dict) or set(row["identity"]) != set(identity):
            raise ValidationError("artifact catalog identity must pin release, benchmark, model and deployment_identity")
        if row["identity"] == identity:
            matches.append(row["artifacts"])
    if len(matches) != 1:
        raise ValidationError("artifact catalog has " + ("no exact match" if not matches else "ambiguous exact matches")
                              + "; provide this model/benchmark's artifact, without fitting on test data")
    artifacts = matches[0]
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValidationError("matching artifact catalog entry has no artifacts")
    from agentinputlab.configuration_assets import artifact_parameters
    base = Path(source).parent
    references = {}
    for name, value in artifacts.items():
        if not isinstance(value, dict) or not isinstance(value.get("path"), str):
            raise ValidationError("catalog artifact references require an explicit path")
        references[name] = {**value, "path": str(base / Path(value["path"]).expanduser())}
    artifact_parameters(workflow.run.release_id, references, load=False)
    resolved = replace(workflow, run=replace(workflow.run, artifacts=references))
    return resolved, {"selected": True, "identity": identity, "catalog_path": source,
                      "binding_sha256": content_hash(references), "fit_performed": False, "model_calls": 0}
