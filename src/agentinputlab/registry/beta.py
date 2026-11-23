"""Machine-readable, fail-closed readiness checks for the v0.2 beta gate."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentinputlab.contracts.enums import ProfileKind, ReleaseTier
from agentinputlab.contracts.serialization import content_hash

from .loader import load_registry_file
from .result_manifest import validate_result_table_manifest
from .runtime import AlgorithmRegistry


@dataclass(frozen=True, slots=True)
class BetaGateCheck:
    """One independently auditable beta release criterion."""

    check_id: str
    passed: bool
    detail: str
    blockers: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "status": "PASS" if self.passed else "BLOCKED",
            "passed": self.passed,
            "detail": self.detail,
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True, slots=True)
class BetaReadinessReport:
    """Deterministic summary suitable for CLI output and release artifacts."""

    gate_id: str
    manifest_sha256: str
    checks: tuple[BetaGateCheck, ...]
    schema_version: int = 1

    @property
    def ready(self) -> bool:
        return all(item.passed for item in self.checks)

    @property
    def blockers(self) -> tuple[str, ...]:
        return tuple(
            f"{item.check_id}: {blocker}" for item in self.checks for blocker in item.blockers
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "gate_id": self.gate_id,
            "status": "READY" if self.ready else "BLOCKED",
            "ready": self.ready,
            "manifest_sha256": self.manifest_sha256,
            "checks": [item.to_dict() for item in self.checks],
            "blockers": list(self.blockers),
        }


def _blocked(check_id: str, detail: str, *blockers: str) -> BetaGateCheck:
    return BetaGateCheck(
        check_id=check_id,
        passed=False,
        detail=detail,
        blockers=tuple(blockers) or ("criterion was not satisfied",),
    )


def _passed(check_id: str, detail: str) -> BetaGateCheck:
    return BetaGateCheck(check_id=check_id, passed=True, detail=detail)


def _repository_path(
    root: Path,
    value: object,
    label: str,
    *,
    reject_symlinks: bool = False,
) -> tuple[Path | None, str | None]:
    if not isinstance(value, str) or not value.strip():
        return None, f"{label} must be a non-empty repository-relative path"
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        return None, f"{label} must be a repository-relative path without parent traversal"
    candidate = root / relative
    if reject_symlinks:
        current = root
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                return None, f"{label} must not contain a symbolic link: {current}"
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None, f"{label} escapes the repository root"
    return candidate, None


def _check_roster(
    registry: AlgorithmRegistry,
    manifest: Mapping[str, Any],
) -> BetaGateCheck:
    expected = manifest.get("required_roster")
    if not isinstance(expected, Mapping):
        return _blocked("roster.20_plus_7", "Frozen algorithm roster", "required_roster missing")
    core_expected = expected.get("core")
    optional_expected = expected.get("optional")
    methods = registry.document.algorithms
    core_actual = sum(item.release_tier is ReleaseTier.CORE for item in methods)
    optional_actual = sum(item.release_tier is ReleaseTier.OPTIONAL for item in methods)
    blockers: list[str] = []
    if core_expected != core_actual:
        blockers.append(f"core roster expected {core_expected!r}, found {core_actual}")
    if optional_expected != optional_actual:
        blockers.append(f"optional roster expected {optional_expected!r}, found {optional_actual}")
    if blockers:
        return _blocked("roster.20_plus_7", "Frozen algorithm roster", *blockers)
    return _passed(
        "roster.20_plus_7",
        f"registry contains {core_actual} core and {optional_actual} optional methods",
    )


def _check_l2_paths(
    root: Path,
    registry: AlgorithmRegistry,
    manifest: Mapping[str, Any],
) -> BetaGateCheck:
    raw_paths = manifest.get("l2_paths")
    minimum = manifest.get("minimum_l2_methods_per_family")
    raw_required_families = manifest.get("required_l2_families")
    if not isinstance(raw_paths, list) or not raw_paths:
        return _blocked(
            "l2.family_coverage",
            "Resolvable documented L2 paths",
            "l2_paths must be a non-empty array",
        )
    if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 1:
        return _blocked(
            "l2.family_coverage",
            "Resolvable documented L2 paths",
            "minimum_l2_methods_per_family must be a positive integer",
        )
    if (
        not isinstance(raw_required_families, list)
        or not raw_required_families
        or any(not isinstance(item, str) or not item for item in raw_required_families)
    ):
        return _blocked(
            "l2.family_coverage",
            "Resolvable documented L2 paths",
            "required_l2_families must be a non-empty array of family names",
        )
    required_families = list(dict.fromkeys(raw_required_families))
    if len(required_families) != len(raw_required_families):
        return _blocked(
            "l2.family_coverage",
            "Resolvable documented L2 paths",
            "required_l2_families must not contain duplicates",
        )

    blockers: list[str] = []
    declared_methods: list[str] = []
    family_counts: Counter[str] = Counter()
    registry_method_ids = {item.method_id for item in registry.document.algorithms}
    for index, raw in enumerate(raw_paths):
        label = f"l2_paths[{index}]"
        if not isinstance(raw, Mapping):
            blockers.append(f"{label} must be an object")
            continue
        method_id = raw.get("method_id")
        release_id = raw.get("release_id")
        if not isinstance(method_id, str) or not method_id:
            blockers.append(f"{label}.method_id must be non-empty")
            continue
        declared_methods.append(method_id)
        if raw.get("evidence_level") != "L2":
            blockers.append(f"{label}.evidence_level must equal L2")
        evidence_path, path_error = _repository_path(
            root,
            raw.get("evidence_path"),
            f"{label}.evidence_path",
        )
        if path_error is not None:
            blockers.append(path_error)
        elif evidence_path is not None and not evidence_path.is_file():
            blockers.append(f"{label}.evidence_path does not exist")
        if not isinstance(release_id, str) or not release_id:
            blockers.append(f"{label}.release_id must be non-empty")
            continue
        try:
            method, release = registry.get_release(release_id)
        except Exception as exc:
            blockers.append(f"{label} references unknown release {release_id!r}: {exc}")
            continue
        if method.method_id != method_id:
            blockers.append(
                f"{label} release belongs to {method.method_id}, not declared {method_id}"
            )
            continue
        if not release.entrypoint_available:
            blockers.append(f"{label} release has no integration entrypoint")
            continue
        try:
            resolved = registry.resolve_entrypoint(release_id)
        except Exception as exc:
            blockers.append(f"{label} entrypoint cannot resolve: {exc}")
            continue
        if not callable(resolved):
            blockers.append(f"{label} entrypoint is not callable")
            continue
        family_counts[method.semantic_family.value] += 1

    duplicates = sorted(
        method_id for method_id, count in Counter(declared_methods).items() if count > 1
    )
    if duplicates:
        blockers.append(f"duplicate L2 method declarations: {', '.join(duplicates)}")
    missing = sorted(registry_method_ids - set(declared_methods))
    extra = sorted(set(declared_methods) - registry_method_ids)
    if missing:
        blockers.append(f"L2 manifest omits methods: {', '.join(missing)}")
    if extra:
        blockers.append(f"L2 manifest declares unknown methods: {', '.join(extra)}")

    registry_families = {item.semantic_family.value for item in registry.document.algorithms}
    unknown_required_families = sorted(set(required_families) - registry_families)
    if unknown_required_families:
        blockers.append(
            "required_l2_families contains unknown families: "
            + ", ".join(unknown_required_families)
        )
    for family in required_families:
        count = family_counts[family]
        if count < minimum:
            blockers.append(
                f"family {family} has {count} resolvable L2 methods; requires at least {minimum}"
            )
    if blockers:
        return _blocked(
            "l2.family_coverage",
            "Resolvable documented L2 paths",
            *blockers,
        )
    rendered = ", ".join(f"{family}={family_counts[family]}" for family in required_families)
    return _passed(
        "l2.family_coverage",
        f"{len(declared_methods)} methods have resolvable L2 paths; {rendered}",
    )


def _check_selectors(manifest: Mapping[str, Any]) -> BetaGateCheck:
    minimum = manifest.get("minimum_selector_count")
    if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 1:
        return _blocked(
            "selector.catalog",
            "Selector catalog size",
            "minimum_selector_count must be a positive integer",
        )
    try:
        from agentinputlab.selectors import list_selector_cards

        cards = list_selector_cards()
    except Exception as exc:
        return _blocked(
            "selector.catalog",
            "Selector catalog size",
            f"selector catalog could not be loaded: {exc}",
        )
    if len(cards) < minimum:
        return _blocked(
            "selector.catalog",
            "Selector catalog size",
            f"found {len(cards)} selectors; requires at least {minimum}",
        )
    return _passed(
        "selector.catalog",
        f"{len(cards)} selector cards are registered (minimum {minimum})",
    )


def _check_algorithm_licenses(registry: AlgorithmRegistry) -> BetaGateCheck:
    blockers: list[str] = []
    for method in registry.document.algorithms:
        upstream = method.upstream
        if not upstream.license_id.strip():
            blockers.append(f"{method.method_id} has no source license identity")
        if upstream.license_id == "NOASSERTION" and upstream.vendoring_allowed:
            blockers.append(f"{method.method_id} may not vendor a NOASSERTION source")
    if blockers:
        return _blocked(
            "license.algorithm_boundaries",
            "Algorithm source-license boundaries",
            *blockers,
        )
    noassertion_count = sum(
        item.upstream.license_id == "NOASSERTION" for item in registry.document.algorithms
    )
    return _passed(
        "license.algorithm_boundaries",
        f"all source licenses are explicit; {noassertion_count} NOASSERTION sources remain external",
    )


def _validate_benchmark_card(
    raw: Any,
    *,
    expected_cell_id: str,
    registry_cell: Mapping[str, Any],
    label: str,
) -> tuple[str, ...]:
    if not isinstance(raw, Mapping):
        return (f"{label} must contain an object",)
    blockers: list[str] = []
    if raw.get("schema_version") != 1:
        blockers.append(f"{label}.schema_version must equal 1")
    if raw.get("cell_id") != expected_cell_id:
        blockers.append(f"{label}.cell_id must equal {expected_cell_id}")
    for key in ("source_url", "source_revision", "license_id"):
        value = raw.get(key)
        if not isinstance(value, str) or not value.strip():
            blockers.append(f"{label}.{key} must be a non-empty string")
    if raw.get("source_revision") != registry_cell.get("source_revision"):
        blockers.append(f"{label}.source_revision does not match benchmark registry")
    vendoring_allowed = raw.get("vendoring_allowed")
    if not isinstance(vendoring_allowed, bool):
        blockers.append(f"{label}.vendoring_allowed must be a boolean")
    if raw.get("license_id") == "NOASSERTION" and vendoring_allowed is not False:
        blockers.append(f"{label} cannot vendor a NOASSERTION source")

    scorer = raw.get("scorer")
    if not isinstance(scorer, Mapping):
        blockers.append(f"{label}.scorer must be an object")
    else:
        for key in ("identity", "revision"):
            value = scorer.get(key)
            if not isinstance(value, str) or not value.strip():
                blockers.append(f"{label}.scorer.{key} must be a non-empty string")

    gold = raw.get("gold_boundary")
    if not isinstance(gold, Mapping):
        blockers.append(f"{label}.gold_boundary must be an object")
    else:
        if gold.get("optimizer_visible") is not False:
            blockers.append(f"{label}.gold_boundary.optimizer_visible must be false")
        if gold.get("evaluator_only") is not True:
            blockers.append(f"{label}.gold_boundary.evaluator_only must be true")
    return tuple(blockers)


def _check_benchmark_cards(
    root: Path,
    manifest: Mapping[str, Any],
) -> BetaGateCheck:
    registry_path, path_error = _repository_path(
        root,
        manifest.get("benchmark_registry"),
        "benchmark_registry",
    )
    if path_error is not None or registry_path is None:
        return _blocked(
            "benchmark.cards",
            "Benchmark source/license/scorer/gold cards",
            path_error or "benchmark_registry is invalid",
        )
    try:
        registry_raw = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return _blocked(
            "benchmark.cards",
            "Benchmark source/license/scorer/gold cards",
            f"cannot load benchmark registry: {exc}",
        )
    raw_cells = registry_raw.get("cells") if isinstance(registry_raw, Mapping) else None
    if not isinstance(raw_cells, list):
        return _blocked(
            "benchmark.cards",
            "Benchmark source/license/scorer/gold cards",
            "benchmark registry cells must be an array",
        )
    cells = {
        item.get("cell_id"): item
        for item in raw_cells
        if isinstance(item, Mapping) and isinstance(item.get("cell_id"), str)
    }

    requirements = manifest.get("required_benchmark_cards")
    if not isinstance(requirements, list) or not requirements:
        return _blocked(
            "benchmark.cards",
            "Benchmark source/license/scorer/gold cards",
            "required_benchmark_cards must be a non-empty array",
        )
    blockers: list[str] = []
    declared: list[str] = []
    for index, requirement in enumerate(requirements):
        label = f"required_benchmark_cards[{index}]"
        if not isinstance(requirement, Mapping):
            blockers.append(f"{label} must be an object")
            continue
        cell_id = requirement.get("cell_id")
        if not isinstance(cell_id, str) or not cell_id:
            blockers.append(f"{label}.cell_id must be non-empty")
            continue
        declared.append(cell_id)
        registry_cell = cells.get(cell_id)
        if registry_cell is None:
            blockers.append(f"{cell_id} is absent from the benchmark registry")
            continue
        card_path, card_error = _repository_path(
            root,
            requirement.get("path"),
            f"{label}.path",
        )
        if card_error is not None or card_path is None:
            blockers.append(card_error or f"{label}.path is invalid")
            continue
        if not card_path.is_file():
            blockers.append(f"{cell_id} benchmark card is missing at {card_path.relative_to(root)}")
            continue
        try:
            card = json.loads(card_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            blockers.append(f"{cell_id} benchmark card cannot be loaded: {exc}")
            continue
        blockers.extend(
            _validate_benchmark_card(
                card,
                expected_cell_id=cell_id,
                registry_cell=registry_cell,
                label=f"benchmark card {cell_id}",
            )
        )
    duplicates = sorted(cell_id for cell_id, count in Counter(declared).items() if count > 1)
    if duplicates:
        blockers.append(f"duplicate benchmark card requirements: {', '.join(duplicates)}")
    if blockers:
        return _blocked(
            "benchmark.cards",
            "Benchmark source/license/scorer/gold cards",
            *blockers,
        )
    return _passed(
        "benchmark.cards",
        f"{len(declared)} required benchmark cards have frozen source, license, scorer, and gold boundaries",
    )


def _check_result_profile_policy(
    root: Path,
    manifest: Mapping[str, Any],
) -> BetaGateCheck:
    policy = manifest.get("result_table_policy")
    if not isinstance(policy, Mapping):
        return _blocked(
            "results.profile_boundary",
            "Single-profile-kind result tables",
            "result_table_policy missing",
        )
    blockers: list[str] = []
    if policy.get("require_single_profile_kind") is not True:
        blockers.append("result_table_policy.require_single_profile_kind must be true")
    allowed = policy.get("allowed_profile_kinds")
    expected = {item.value for item in ProfileKind}
    if not isinstance(allowed, list) or set(allowed) != expected:
        blockers.append(
            "result_table_policy.allowed_profile_kinds must contain exactly "
            + ", ".join(sorted(expected))
        )
    if policy.get("validator") != "agentinputlab.registry:validate_result_table_manifest":
        blockers.append("result_table_policy.validator must name the shipped validator")
    minimum_manifests = policy.get("minimum_manifests")
    if (
        isinstance(minimum_manifests, bool)
        or not isinstance(minimum_manifests, int)
        or minimum_manifests < 1
    ):
        blockers.append("result_table_policy.minimum_manifests must be a positive integer")
        minimum_manifests = 1
    manifest_glob = policy.get("manifest_glob")
    if not isinstance(manifest_glob, str) or not manifest_glob.strip():
        blockers.append("result_table_policy.manifest_glob must be non-empty")
        result_paths: list[Path] = []
    elif Path(manifest_glob).is_absolute() or ".." in Path(manifest_glob).parts:
        blockers.append("result_table_policy.manifest_glob must remain repository-relative")
        result_paths = []
    else:
        result_paths = sorted(root.glob(manifest_glob))
        if len(result_paths) < minimum_manifests:
            blockers.append(
                "result-table manifest inventory is incomplete: "
                f"found {len(result_paths)}, requires at least {minimum_manifests}"
            )
        seen_table_ids: dict[str, Path] = {}
        seen_run_identities: dict[str, Path] = {}
        for path in result_paths:
            try:
                relative_manifest_path = path.relative_to(root)
            except ValueError:
                blockers.append(f"result-table manifest path escapes repository root: {path}")
                continue
            checked_path, checked_path_error = _repository_path(
                root,
                relative_manifest_path.as_posix(),
                f"result-table manifest {relative_manifest_path}",
                reject_symlinks=True,
            )
            if checked_path_error is not None or checked_path is None:
                blockers.append(
                    checked_path_error
                    or f"result-table manifest {relative_manifest_path} is invalid"
                )
                continue
            if not checked_path.is_file():
                blockers.append(
                    f"result-table manifest {relative_manifest_path} is not a regular file"
                )
                continue
            try:
                manifest_bytes = checked_path.read_bytes()
                raw = json.loads(manifest_bytes)
                table = validate_result_table_manifest(raw)
            except Exception as exc:
                blockers.append(f"{relative_manifest_path} violates profile boundary: {exc}")
                continue
            previous_table = seen_table_ids.get(table.table_id)
            if previous_table is not None:
                blockers.append(
                    f"{relative_manifest_path} duplicates table_id {table.table_id!r} "
                    f"already declared by {previous_table}"
                )
            else:
                seen_table_ids[table.table_id] = relative_manifest_path
            for study in table.studies:
                previous_run = seen_run_identities.get(study.run_identity_hash)
                if previous_run is not None:
                    blockers.append(
                        f"{relative_manifest_path} duplicates run_identity_hash "
                        f"{study.run_identity_hash} already referenced by {previous_run}"
                    )
                else:
                    seen_run_identities[study.run_identity_hash] = relative_manifest_path
                run_path, run_path_error = _repository_path(
                    root,
                    study.run_path,
                    f"{relative_manifest_path} study {study.study_id} run_path",
                    reject_symlinks=True,
                )
                if run_path_error is not None or run_path is None:
                    blockers.append(run_path_error or f"{study.study_id} run_path is invalid")
                    continue
                try:
                    from agentinputlab.experiments import verify_experiment_run_directory

                    verified = verify_experiment_run_directory(run_path)
                except Exception as exc:
                    blockers.append(
                        f"{relative_manifest_path} study {study.study_id} "
                        f"does not reference a verified run: {exc}"
                    )
                    continue
                expected = {
                    "study_id": study.study_id,
                    "profile_kind": study.profile_kind.value,
                    "comparison_profile_hash": study.profile_hash,
                    "identity_hash": study.run_identity_hash,
                    "manifest_hash": study.manifest_hash,
                    "plan_hash": study.plan_hash,
                    "driver_hash": study.driver_hash,
                }
                mismatches = [key for key, value in expected.items() if verified.get(key) != value]
                if mismatches:
                    blockers.append(
                        f"{relative_manifest_path} study {study.study_id} verified "
                        "run identity differs for: " + ", ".join(sorted(mismatches))
                    )
                if not verified.get("complete") or not verified.get("all_succeeded"):
                    blockers.append(
                        f"{relative_manifest_path} study {study.study_id} "
                        "run is incomplete or contains failed cells"
                    )
                if verified.get("is_shard"):
                    blockers.append(
                        f"{relative_manifest_path} study {study.study_id} "
                        "references a shard rather than a complete parent plan"
                    )
    if blockers:
        return _blocked(
            "results.profile_boundary",
            "Single-profile-kind result tables",
            *blockers,
        )
    return _passed(
        "results.profile_boundary",
        f"profile-kind policy is enforced; {len(result_paths)} result-table manifests validated",
    )


def check_beta_readiness(
    root: str | Path,
    *,
    manifest_path: str | Path | None = None,
    registry: AlgorithmRegistry | None = None,
) -> BetaReadinessReport:
    """Evaluate integrity-oriented beta gates without claiming execution authenticity."""

    supplied_root = Path(root).expanduser().absolute()
    repository_root = supplied_root.resolve()
    supplied_manifest = (
        Path(manifest_path).expanduser()
        if manifest_path is not None
        else Path("manifests") / "releases" / "v0.2-beta.json"
    )
    if supplied_manifest.is_absolute():
        try:
            relative_manifest = supplied_manifest.relative_to(supplied_root)
        except ValueError:
            try:
                relative_manifest = supplied_manifest.relative_to(repository_root)
            except ValueError:
                relative_manifest = None
    else:
        relative_manifest = supplied_manifest
    if relative_manifest is None:
        selected_manifest = supplied_manifest
        selected_manifest_error = "beta release manifest escapes the repository root"
    else:
        selected_manifest, selected_manifest_error = _repository_path(
            repository_root,
            relative_manifest.as_posix(),
            "beta release manifest",
            reject_symlinks=True,
        )
    if selected_manifest_error is not None or selected_manifest is None:
        return BetaReadinessReport(
            gate_id="v0.2-beta",
            manifest_sha256="",
            checks=(
                _blocked(
                    "manifest.schema",
                    "Beta release manifest",
                    selected_manifest_error or "beta release manifest path is invalid",
                ),
            ),
        )
    try:
        manifest_bytes = selected_manifest.read_bytes()
        raw = json.loads(manifest_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        return BetaReadinessReport(
            gate_id="v0.2-beta",
            manifest_sha256="",
            checks=(
                _blocked(
                    "manifest.schema",
                    "Beta release manifest",
                    f"cannot load {selected_manifest}: {exc}",
                ),
            ),
        )
    if not isinstance(raw, Mapping):
        return BetaReadinessReport(
            gate_id="v0.2-beta",
            manifest_sha256=content_hash(raw),
            checks=(
                _blocked(
                    "manifest.schema",
                    "Beta release manifest",
                    "manifest root must be an object",
                ),
            ),
        )
    gate_id = raw.get("gate_id")
    schema_version = raw.get("schema_version")
    manifest_blockers: list[str] = []
    if schema_version != 1:
        manifest_blockers.append("schema_version must equal 1")
    if gate_id != "v0.2-beta":
        manifest_blockers.append("gate_id must equal v0.2-beta")
    if manifest_blockers:
        return BetaReadinessReport(
            gate_id=gate_id if isinstance(gate_id, str) else "v0.2-beta",
            manifest_sha256=content_hash(raw),
            checks=(
                _blocked(
                    "manifest.schema",
                    "Beta release manifest",
                    *manifest_blockers,
                ),
            ),
        )

    active_registry = registry
    if active_registry is None:
        source_registry = repository_root / "manifests" / "algorithms" / "registry.yaml"
        active_registry = AlgorithmRegistry(load_registry_file(source_registry))
    checks = (
        _check_roster(active_registry, raw),
        _check_l2_paths(repository_root, active_registry, raw),
        _check_selectors(raw),
        _check_algorithm_licenses(active_registry),
        _check_benchmark_cards(repository_root, raw),
        _check_result_profile_policy(repository_root, raw),
    )
    return BetaReadinessReport(
        gate_id=gate_id,
        manifest_sha256=content_hash(raw),
        checks=checks,
    )


__all__ = [
    "BetaGateCheck",
    "BetaReadinessReport",
    "check_beta_readiness",
]
