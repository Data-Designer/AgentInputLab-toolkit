"""Dependency-free AgentInputLab command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from agentinputlab.contracts.enums import ReleaseTier, SemanticFamily, SupportStatus
from agentinputlab.contracts.errors import AgentInputLabError, RegistryError
from agentinputlab.contracts.serialization import to_jsonable
from agentinputlab.registry import (
    AlgorithmCard,
    AlgorithmRegistry,
    find_repository_registry,
    load_packaged_registry,
    load_registry_file,
    validate_snapshot_sync,
)


def _cmd_smoke(args: argparse.Namespace) -> int:
    """Run the dependency-free Query2doc to BM25 reference loop."""

    from agentinputlab import FrozenAgentState, InterventionContext
    from agentinputlab.algorithms.reformulate.query2doc import (
        OfficialQuery2DocRow,
        Query2DocCache,
        Query2DocFixtureCached,
    )
    from agentinputlab.backends.retrieval import Document, InMemoryBM25Retriever
    from agentinputlab.runner import InterventionRunner

    state = FrozenAgentState(task_id="tiny-q1", query="capital france")
    cache = Query2DocCache(
        (
            OfficialQuery2DocRow(
                query_id="tiny-q1",
                query=state.query,
                pseudo_doc="Paris is the capital of France.",
                split="tiny",
            ),
        )
    )
    algorithm = Query2DocFixtureCached(cache, split="tiny")
    record = InterventionRunner().run(
        algorithm,
        state,
        InterventionContext(seed=0),
    )
    transformed = record.patch.apply(state)
    retriever = InMemoryBM25Retriever(
        (
            Document(
                doc_id="fr",
                text="Paris is the capital of France.",
                corpus_revision="synthetic-v1",
            ),
            Document(
                doc_id="de",
                text="Berlin is the capital of Germany.",
                corpus_revision="synthetic-v1",
            ),
        )
    )
    results = retriever.retrieve(transformed.query, top_k=2)
    payload = {
        "ok": bool(results and results[0].document.doc_id == "fr"),
        "offline": True,
        "profile_kind": record.patch.profile_kind,
        "release_id": algorithm.release_id,
        "input_state_hash": record.input_state_hash,
        "patch_hash": record.patch.patch_hash,
        "top_document_id": results[0].document.doc_id if results else None,
    }
    if args.json:
        _json_dump(payload)
    else:
        print(
            "offline smoke: "
            f"{'OK' if payload['ok'] else 'FAILED'} - "
            f"{payload['release_id']} -> {payload['top_document_id']}"
        )
    return 0 if payload["ok"] else 1


def _json_dump(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _cmd_docs(args: argparse.Namespace) -> int:
    from agentinputlab.documentation import list_guides, read_guide

    if args.list:
        topics = list_guides()
        if args.json:
            _json_dump({"topics": topics, "offline": True})
        else:
            print("\n".join(topics))
    else:
        content = read_guide(args.topic)
        if args.json:
            _json_dump({"topic": args.topic, "content": content, "offline": True})
        else:
            print(content, end="" if content.endswith("\n") else "\n")
    return 0


def _cmd_config(args: argparse.Namespace) -> int:
    from agentinputlab.configuration import RunConfig, load_run_state

    if args.config_action == "init":
        from agentinputlab.api import describe_intervention

        describe_intervention(args.release_id)
        # stdout only: the user chooses where to save and no file is overwritten.
        _json_dump(RunConfig(args.release_id).to_dict())
        return 0
    config = RunConfig.from_file(args.path)
    if args.config_action == "run":
        from agentinputlab.backends.llm.openai_compatible import OpenAICompatibleTransportError
        from agentinputlab.contracts.errors import ExternalDependencyError

        try:
            result = config.run(load_run_state(args.state) if args.state else args.query)
        except OpenAICompatibleTransportError as exc:
            # This type's message is deliberately sanitized by the backend.
            raise ExternalDependencyError(str(exc)) from None
        except RuntimeError:
            # Native providers may put credentials in exception text.
            raise ExternalDependencyError(
                "configured runtime failed; check endpoint/schema and native prerequisites; "
                "no automatic retry (the request may have consumed tokens)"
            ) from None
    else:
        result = config.validate()
    _json_dump(result)
    return 0


def _cmd_prepare_adaptive(args: argparse.Namespace) -> int:
    from agentinputlab.adaptive_configuration import prepare
    from agentinputlab.configuration import RunConfig

    _json_dump(prepare(RunConfig.from_file(args.path), args.output_dir,
                      router_checkpoint=args.router_checkpoint))
    return 0


def _cmd_prepare_qrag(args: argparse.Namespace) -> int:
    from agentinputlab.configuration import RunConfig, load_run_state
    from agentinputlab.qrag_configuration import prepare

    _json_dump(prepare(RunConfig.from_file(args.path), load_run_state(args.state), args.output_dir,
                      reader_repository=args.reader_repository, reader_revision=args.reader_revision,
                      reader_profile=args.reader_profile))
    return 0


def _cmd_prepare_longllmlingua(args: argparse.Namespace) -> int:
    from agentinputlab.configuration import RunConfig
    from agentinputlab.longllmlingua_configuration import prepare

    _json_dump(prepare(RunConfig.from_file(args.path), args.output_dir))
    return 0


def _cmd_prepare_recomp(args: argparse.Namespace) -> int:
    from agentinputlab.configuration import RunConfig
    from agentinputlab.recomp_configuration import prepare

    _json_dump(prepare(RunConfig.from_file(args.path), args.output_dir))
    return 0


def _cmd_prepare_worker(args: argparse.Namespace) -> int:
    from agentinputlab.configuration import RunConfig
    from agentinputlab.worker_preparation import prepare

    _json_dump(prepare(RunConfig.from_file(args.path), args.output_dir))
    return 0


def _cmd_prepare_hyde(args: argparse.Namespace) -> int:
    from agentinputlab.configuration import RunConfig
    from agentinputlab.hyde_configuration import prepare

    _json_dump(prepare(RunConfig.from_file(args.path), args.output_dir))
    return 0


def _cmd_prepare_query2doc(args: argparse.Namespace) -> int:
    from agentinputlab.configuration import RunConfig
    from agentinputlab.query2doc_configuration import prepare

    _json_dump(prepare(RunConfig.from_file(args.path), args.output_dir))
    return 0


def _cmd_prepare_ircot(args: argparse.Namespace) -> int:
    from agentinputlab.configuration import RunConfig
    from agentinputlab.ircot_configuration import prepare

    _json_dump(prepare(RunConfig.from_file(args.path), args.output_dir,
                       accept_substitutions=args.accept_substitutions))
    return 0


def _cmd_config_fit(args: argparse.Namespace) -> int:
    from agentinputlab.contracts import ValidationError
    from agentinputlab.fit_configuration import FitConfig

    config = FitConfig.from_file(args.path)
    if args.execute:
        if not args.output_dir:
            raise ValidationError('config fit --execute requires --output-dir with a new directory')
        result = config.execute(args.output_dir)
    else:
        result = config.validate()
    _json_dump(result)
    return 0


def _cmd_prepare_fit(args: argparse.Namespace) -> int:
    from agentinputlab.fit_preparation import prepare_fit

    _json_dump(prepare_fit(args.path, output_dir=args.output_dir))
    return 0


def _cmd_config_online(args: argparse.Namespace) -> int:
    from agentinputlab.contracts import ValidationError
    from agentinputlab.online_configuration import OnlineConfig

    config = OnlineConfig.from_file(args.path)
    if args.execute:
        if not args.output_dir:
            raise ValidationError('config online --execute requires --output-dir with a new directory')
        result = config.execute(args.output_dir)
    else:
        result = config.validate()
    _json_dump(result)
    return 2 if result.get('status') in ('native_failed_terminal', 'execution_failed_before_model') else 0


def _cmd_config_evaluate(args: argparse.Namespace) -> int:
    from contextlib import ExitStack
    from pathlib import Path

    from agentinputlab.configuration import RunConfig, _json_mapping, _unique
    from agentinputlab.contracts.errors import ValidationError

    config = RunConfig.from_file(args.path)
    predictor = None
    qrag_reader = None
    longbench_sources = None
    beir_sample_sources = None
    trec_sample_sources = None
    if getattr(args, "trec_sample_config", None):
        from agentinputlab.trec_sample_configuration import read_sources

        trec_sample_sources = read_sources(args.trec_sample_config)
    if getattr(args, "beir_sample_config", None):
        from agentinputlab.beir_sample_configuration import read_sources

        beir_sample_sources = read_sources(args.beir_sample_config)
    if getattr(args, "longbench_config", None):
        from agentinputlab.longbench_configuration import read_sources

        longbench_sources = read_sources(args.longbench_config)
    if getattr(args, "qrag_reader_config", None):
        path = Path(args.qrag_reader_config).absolute()
        try:
            with path.open("rb") as source:
                raw = source.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024:
                raise ValidationError("Q-RAG reader configuration exceeds 1 MiB")
            qrag_reader = dict(_json_mapping(json.loads(raw, object_pairs_hook=_unique), "Q-RAG reader"))
            if isinstance(qrag_reader.get("policy_replays"), (tuple, list)):
                qrag_reader["policy_replays"] = [
                    {**ref, "path": str(path.parent / ref["path"])}
                    if isinstance(ref, dict) and isinstance(ref.get("path"), str) and not Path(ref["path"]).is_absolute()
                    else ref for ref in to_jsonable(qrag_reader["policy_replays"])]
        except (OSError, UnicodeError, json.JSONDecodeError, RecursionError):
            raise ValidationError("cannot read Q-RAG reader configuration: use a UTF-8 JSON object") from None
    if args.predictor_config:
        try:
            with open(args.predictor_config, "rb") as stream:
                raw = stream.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024:
                raise ValidationError("predictor configuration exceeds 1 MiB")
            predictor = _json_mapping(json.loads(raw, object_pairs_hook=_unique), "predictor")
            ranking = predictor.get("archived_retrieval")
            if (isinstance(ranking, Mapping) and isinstance(ranking.get("path"), str)
                    and ranking["path"].strip() and not Path(ranking["path"]).is_absolute()):
                predictor = {**predictor, "archived_retrieval": {
                    **ranking, "path": str(Path(args.predictor_config).absolute().parent / ranking["path"])}}
        except (OSError, UnicodeError, json.JSONDecodeError, RecursionError):
            raise ValidationError("cannot read predictor configuration: use a UTF-8 JSON object") from None
    with ExitStack() as stack:
        try:
            stream = stack.enter_context(open(args.output, "x", encoding="utf-8"))
        except OSError:
            raise ValidationError("output must be a new writable JSONL path; existing evidence is never overwritten") from None
        def persist(record):
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()

        result = config.evaluate(
            args.benchmark, split=args.split, example_ids=args.case_id, limit=args.limit,
            prepared_root=args.prepared_root, cache_dir=args.cache_dir, offline=args.offline,
            accept_license_review=args.accept_license_review,
            predictor_parameters=predictor, on_record=persist,
            qrag_reader=qrag_reader,
            bfcl_source_archive=args.bfcl_source_archive,
            longbench_sources=longbench_sources,
            beir_sample_sources=beir_sample_sources,
            trec_sample_sources=trec_sample_sources,
        )
        summary = {key: value for key, value in result.items() if key != "records"}
        persist({"status": "batch_complete", **summary})
    _json_dump({**summary, "records_path": str(args.output)})
    return 0 if result["failed_count"] == 0 and result["awaiting_clarification_count"] == 0 else 1


def _cmd_config_arena(args: argparse.Namespace) -> int:
    from agentinputlab.configuration import RunConfig

    result = RunConfig.from_file(args.path).evaluate_arena(
        args.arena_config, output_dir=args.output_dir,
        case_ids=args.case_id, limit=args.limit, inventory_only=args.list_cases, capture_only=args.capture_only)
    _json_dump(result)
    return 0 if result["failed_count"] == 0 else 1


def _cmd_prepare_arena(args: argparse.Namespace) -> int:
    from agentinputlab.arena_preparation import prepare_arena

    _json_dump(prepare_arena(args.path, environment=args.environment, python=args.python,
                            allow_network=args.allow_network, toolkit_wheel=args.toolkit_wheel))
    return 0


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    lines = [
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    lines.extend(
        "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)) for row in rows
    )
    return "\n".join(lines)


def _registry_from_argument(path: str | None) -> AlgorithmRegistry:
    document = load_registry_file(path) if path else load_packaged_registry()
    return AlgorithmRegistry(document)


def _cmd_list(args: argparse.Namespace) -> int:
    registry = _registry_from_argument(args.registry)
    if args.releases:
        records = registry.list_releases(
            family=args.family,
            tier=args.tier,
            status=args.status,
            delivery_tier=args.delivery_tier,
        )
        records = tuple(
            (method, release) for method, release in records
            if args.search.casefold() in
            f"{method.method_id} {method.display_name} {release.id}".casefold()
        )
        if args.json:
            _json_dump(
                [
                    {
                        "method_id": method.method_id,
                        "display_name": method.display_name,
                        "semantic_family": method.semantic_family.value,
                        "release_tier": method.release_tier.value,
                        **release.to_dict(),
                        "entrypoint_available": release.entrypoint_available,
                        "runnable": release.runnable,
                        "availability": release.availability,
                        "blocker": release.blocker,
                    }
                    for method, release in records
                ]
            )
            return 0
        rows = [
            (
                release.id,
                method.display_name,
                method.semantic_family.value,
                release.status.value,
                release.delivery_tier,
                "yes" if release.entrypoint_available else "no",
                "yes" if release.runnable else "no",
                release.availability,
                release.blocker,
            )
            for method, release in records
        ]
        print(
            _table(
                (
                    "RELEASE",
                    "METHOD",
                    "FAMILY",
                    "STATUS",
                    "DELIVERY",
                    "ENTRYPOINT",
                    "RUNNABLE",
                    "AVAILABILITY",
                    "BLOCKER",
                ),
                rows,
            )
        )
        return 0

    methods = registry.list_methods(family=args.family, tier=args.tier, status=args.status)
    methods = tuple(
        method for method in methods
        if args.search.casefold() in f"{method.method_id} {method.display_name}".casefold()
    )
    if args.json:
        _json_dump([method.to_dict() for method in methods])
        return 0
    rows = [
        (
            method.method_id,
            method.display_name,
            method.semantic_family.value,
            method.release_tier.value,
            method.status.value,
            str(len(method.releases)),
        )
        for method in methods
    ]
    print(_table(("METHOD", "NAME", "FAMILY", "TIER", "STATUS", "RELEASES"), rows))
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    registry = _registry_from_argument(args.registry)
    if args.parameters:
        from agentinputlab.api import describe_intervention

        _json_dump(describe_intervention(args.identity, registry=registry))
        return 0
    result = registry.get(args.identity)
    if isinstance(result, AlgorithmCard):
        payload = result.to_dict()
    else:
        method, release = result
        payload = {"method": method.to_dict(), "release": release.to_dict()}
    _json_dump(payload)
    return 0


def _cmd_dataset_list(args: argparse.Namespace) -> int:
    from agentinputlab.benchmarks import list_benchmarks

    records = list_benchmarks(search=args.search, family=args.family)
    if args.json:
        _json_dump(records)
    else:
        print(_table(("DATASET", "TASK", "LOADER", "SOURCE MiB"), [
            (row["cell_id"], row["task"], row["loader_kind"],
             "—" if row["source_download_bytes"] is None
             else f"{row['source_download_bytes'] / 1024**2:.1f}")
            for row in records
        ]))
        print("Inspect prerequisites: agentinputlab datasets show DATASET")
    return 0


def _cmd_dataset_show(args: argparse.Namespace) -> int:
    from agentinputlab.benchmarks import describe_benchmark

    _json_dump(describe_benchmark(args.cell_id))
    return 0


def _cmd_validate_registry(args: argparse.Namespace) -> int:
    if args.path:
        document = load_registry_file(args.path)
        source = Path(args.path).expanduser().resolve()
    else:
        repository_source = find_repository_registry()
        if repository_source is None:
            document = load_packaged_registry()
            source = Path("agentinputlab.registry/data/algorithms.json")
        else:
            document = load_registry_file(repository_source)
            source = repository_source
    if args.check_snapshot and source.is_file():
        validate_snapshot_sync(source)
    registry = AlgorithmRegistry(document)
    report = registry.doctor()
    payload = {
        "valid": report.ok,
        "source": str(source),
        **report.to_dict(),
    }
    if args.json:
        _json_dump(payload)
    else:
        print(
            f"registry valid: {source} "
            f"({report.method_count} methods, {report.release_count} releases, "
            f"{report.must_ship_count} must-ship, {report.entrypoint_count} entrypoints, "
            f"{report.locked_count} locked)"
        )
    return 0 if report.ok else 1


def _cmd_doctor(args: argparse.Namespace) -> int:
    registry = _registry_from_argument(args.registry)
    report = registry.doctor()
    source_path = Path(args.source).expanduser().resolve() if args.source else None
    if source_path is None:
        source_path = find_repository_registry()

    sync_ok: bool | None = None
    errors = list(report.errors)
    if source_path is not None:
        try:
            validate_snapshot_sync(source_path)
            sync_ok = True
        except RegistryError as exc:
            sync_ok = False
            errors.append(str(exc))

    ok = report.ok and sync_ok is not False
    payload = {
        **report.to_dict(),
        "ok": ok,
        "offline": bool(args.offline),
        "source_registry": str(source_path) if source_path else None,
        "snapshot_synchronized": sync_ok,
        "errors": errors,
    }
    if args.json:
        _json_dump(payload)
    else:
        state = "OK" if ok else "FAILED"
        print(
            f"registry doctor: {state} — {report.method_count} methods, "
            f"{report.release_count} releases, {report.must_ship_count} must-ship, "
            f"{report.entrypoint_count} entrypoints, {report.locked_count} locked"
        )
        if sync_ok is True:
            print("source snapshot: synchronized")
        elif sync_ok is False:
            print("source snapshot: OUT OF SYNC")
        elif source_path is None:
            print("source snapshot: not checked (installed-package mode)")
        for warning in report.warnings:
            print(f"warning: {warning}")
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
    return 0 if ok else 1


def _cmd_selector_list(args: argparse.Namespace) -> int:
    from agentinputlab.selectors import list_selector_cards

    cards = list_selector_cards()
    payload = [
        {
            "selector_id": card.selector_id,
            "display_name": card.display_name,
            "kind": card.kind.value,
            "requires_fit": card.requires_fit,
            "optional_dependency": card.optional_dependency or None,
        }
        for card in cards
    ]
    if args.json:
        _json_dump(payload)
    else:
        print(
            _table(
                ("SELECTOR", "NAME", "KIND", "REQUIRES_FIT", "OPTIONAL_DEPENDENCY"),
                [
                    (
                        item["selector_id"],
                        item["display_name"],
                        item["kind"],
                        "yes" if item["requires_fit"] else "no",
                        item["optional_dependency"] or "-",
                    )
                    for item in payload
                ],
            )
        )
    return 0


def _cmd_selector_features(args: argparse.Namespace) -> int:
    from agentinputlab.contracts import FrozenAgentState
    from agentinputlab.selectors import VisibleStateFeatureExtractor

    state = FrozenAgentState(
        query=args.query,
        instruction=args.instruction,
        tool_names=tuple(args.tool_name),
    )
    observation = VisibleStateFeatureExtractor().extract(state)
    _json_dump(
        {
            "state_hash": state.state_hash,
            "feature_hash": observation.feature_hash,
            "features": to_jsonable(observation.features),
            "extractor": {
                "id": observation.extractor_id,
                "revision": observation.extractor_revision,
            },
            "access_domains": [item.value for item in observation.access_domains],
            "provenance": to_jsonable(observation.provenance),
            "raw_text_persisted": False,
        }
    )
    return 0


def _cmd_selector_inspect(args: argparse.Namespace) -> int:
    from agentinputlab.selectors import (
        SelectorArtifactIdentity,
        SelectorArtifactStore,
    )

    identity = SelectorArtifactIdentity(
        selector_id=args.selector_id,
        revision=args.revision,
        sha256=args.sha256,
    )
    _selector, bundle = SelectorArtifactStore(args.store).load(identity)
    _json_dump(
        {
            "identity": to_jsonable(identity),
            "stage": bundle.stage.value,
            "bundle_hash": bundle.bundle_hash,
            "model_hash": bundle.model_hash,
            "extractor": {
                "id": bundle.extractor_id,
                "revision": bundle.extractor_revision,
                "sha256": bundle.extractor_hash,
            },
            "feature_schema": {
                "id": bundle.feature_schema_id,
                "sha256": bundle.feature_schema_hash,
            },
            "representative_by_family": {
                family.value: release_id
                for family, release_id in bundle.representative_by_family.items()
            },
            "training_table_sha256": bundle.training_table_sha256,
            "split_provenance_hash": bundle.split_provenance_hash,
            "comparison_profile_hash": bundle.comparison_profile_hash,
        }
    )
    return 0


def _cmd_dataset_prepare(args: argparse.Namespace) -> int:
    from agentinputlab.benchmarks import prepare_benchmark

    _json_dump(
        prepare_benchmark(
            args.cell_id,
            cache_dir=args.cache_dir,
            offline=args.offline,
            accept_license_review=args.accept_license_review,
            max_download_bytes=args.max_download_mib * 1024 * 1024,
            max_prepared_bytes=args.max_prepared_mib * 1024 * 1024,
            source_archive=args.source_archive,
        )
    )
    return 0


def _cmd_trec_sample(args: argparse.Namespace) -> int:
    from agentinputlab.data import prepare_trec_dl_sample

    _json_dump(
        prepare_trec_dl_sample(
            args.root,
            year=args.year,
            query_count=args.query_count,
            query_offset=args.query_offset,
            offline=args.offline,
            accept_license_review=args.accept_license_review,
            max_download_bytes=args.max_download_mib * 1024 * 1024,
            expected_corpus_revision=args.corpus_revision,
            source_sample_root=args.source_sample_root,
            source_manifest_sha256=args.source_manifest_sha256,
        )
    )
    return 0


def _cmd_campaign_doctor(args: argparse.Namespace) -> int:
    from agentinputlab.campaign import campaign_doctor

    readiness = args.readiness
    if readiness is None:
        candidate = Path("configs/parity/adapter_readiness.json")
        readiness = candidate if candidate.is_file() else None
    report = campaign_doctor(
        args.prepared,
        readiness_path=readiness,
    )
    _json_dump(to_jsonable(report))
    target_ready = (
        report["static_data_ready"] if args.target == "static" else report["full_campaign_ready"]
    )
    return 0 if target_ready else 2


def _cmd_campaign_prepare(args: argparse.Namespace) -> int:
    from agentinputlab.benchmarks import (
        BenchmarkSmokeAssetRegistry,
        materialize_benchmark_asset,
    )
    from agentinputlab.campaign import existing_campaign_assets

    registry = BenchmarkSmokeAssetRegistry.default()
    reports: list[dict[str, Any]] = []
    blocked: list[dict[str, str]] = []
    for asset_id in existing_campaign_assets():
        asset = registry.get(asset_id)
        if asset.status == "license_review_required" and not args.accept_license_review:
            blocked.append(
                {
                    "asset_id": asset_id,
                    "reason": (
                        "license review is required; rerun with "
                        "--accept-license-review after reviewing the source terms"
                    ),
                }
            )
            continue
        reports.append(
            materialize_benchmark_asset(
                asset_id,
                store_root=args.store,
                prepared_root=args.prepared,
                allow_network=args.allow_network,
                accept_license_review=args.accept_license_review,
            )
        )
    _json_dump(
        {
            "schema_version": 1,
            "protocol": "agentinputlab.campaign-prepare.v1",
            "scope": "existing-data-main",
            "assets": reports,
            "blocked": blocked,
            "inventory_only_cells_touched": 0,
            "experiment_executed": False,
        }
    )
    return 0 if not blocked else 2


def _cmd_campaign_bootstrap(args: argparse.Namespace) -> int:
    from agentinputlab.runtime_bootstrap import bootstrap_runtime_sources

    report = bootstrap_runtime_sources(
        args.root,
        method_ids=args.method or (() if args.arena else None),
        include_arenas=not args.no_arenas,
        arena_ids=args.arena,
        allow_network=args.allow_network,
    )
    _json_dump(to_jsonable(report))
    return 0 if report["ready_count"] == report["source_count"] else 2


def _cmd_campaign_smoke(args: argparse.Namespace) -> int:
    from agentinputlab.campaign import campaign_data_smoke

    _json_dump(to_jsonable(campaign_data_smoke(args.prepared)))
    return 0


def _cmd_campaign_methods(args: argparse.Namespace) -> int:
    from agentinputlab.campaign import campaign_method_matrix

    report = campaign_method_matrix(args.catalog, args.readiness)
    _json_dump(to_jsonable(report))
    return 0 if report["adapter_ready"] == report["method_count"] else 2


def _cmd_campaign_run(args: argparse.Namespace) -> int:
    from agentinputlab.experiments.cli import main as experiment_main

    arguments = ["run", str(args.manifest)]
    if args.driver:
        arguments.extend(("--driver", args.driver))
    if args.no_resolve_units:
        arguments.append("--no-resolve-units")
    if args.shard_count is not None:
        arguments.extend(("--shard-count", str(args.shard_count)))
        arguments.extend(("--shard-index", str(args.shard_index)))
    if args.execute:
        arguments.append("--execute")
    arguments.extend(("--output-root", str(args.output_root)))
    if args.resume:
        arguments.append("--resume")
    if args.fail_fast:
        arguments.append("--fail-fast")
    return experiment_main(arguments)


def _cmd_campaign_verify(args: argparse.Namespace) -> int:
    from agentinputlab.campaign import campaign_verify_run

    _json_dump(
        to_jsonable(
            campaign_verify_run(
                args.run_dir,
                allow_incomplete=args.allow_incomplete,
            )
        )
    )
    return 0


def _cmd_campaign_analyze(args: argparse.Namespace) -> int:
    from agentinputlab.experiments import analyze_confirmatory_protocol

    result = analyze_confirmatory_protocol(
        tuple(str(path) for path in args.run_dir),
        protocol=args.protocol,
        metric=args.metric,
        algorithm_failure_value=args.algorithm_failure_value,
        bootstrap_samples=args.bootstrap_samples,
        permutation_samples=args.permutation_samples,
        random_seed=args.random_seed,
    )
    _json_dump(to_jsonable(result))
    return 0


def _cmd_campaign_run_all(args: argparse.Namespace) -> int:
    from agentinputlab.campaign_runner import (
        load_one_click_campaign,
        run_one_click_campaign,
    )

    campaign = load_one_click_campaign(args.config)
    result = run_one_click_campaign(campaign, execute=args.execute)
    _json_dump(to_jsonable(result))
    if not args.execute:
        return 0 if result["ready"] else 2
    if result.get("execution_blocked") is True:
        return 2
    return 0 if result["all_succeeded"] else 3


def _cmd_campaign_run_production(args: argparse.Namespace) -> int:
    from agentinputlab.production_campaign import (
        load_production_campaign,
        run_production_campaign,
    )

    campaign = load_production_campaign(args.config)
    result = run_production_campaign(campaign, execute=args.execute)
    _json_dump(to_jsonable(result))
    if not args.execute:
        return 0 if result["ready"] else 2
    if result.get("execution_blocked") is True:
        return 2
    return 0 if result["all_succeeded"] else 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentinputlab",
        description="Discover algorithms and datasets, prepare data, and diagnose AgentInputLab.",
    )
    from agentinputlab import __version__

    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--debug", action="store_true", help="Show tracebacks for toolkit errors.")
    parser.add_argument(
        "--registry",
        help="Use a source registry instead of the packaged snapshot.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    from agentinputlab.workflows.cli import add_parser as add_workflow_parser
    add_workflow_parser(subparsers)

    docs_parser = subparsers.add_parser("docs", help="Read bundled guides offline; no GitHub account needed.")
    docs_parser.add_argument("topic", nargs="?", default="getting-started")
    docs_parser.add_argument("--list", action="store_true", help="List exact documentation topics.")
    docs_parser.add_argument("--json", action="store_true")
    docs_parser.set_defaults(func=_cmd_docs)

    config_parser = subparsers.add_parser("config", help="Create, validate or run an editable JSON config.")
    config_commands = config_parser.add_subparsers(dest="config_action", required=True)
    config_init = config_commands.add_parser("init", help="Print a template; fill native parameters before running.")
    config_init.add_argument("--release-id", default="plan.plan_and_solve.ps_prompt")
    config_init.set_defaults(func=_cmd_config)
    config_validate = config_commands.add_parser("validate", help="Check JSON/parameters without model calls.")
    config_validate.add_argument("path")
    config_validate.set_defaults(func=_cmd_config)
    config_run = config_commands.add_parser("run", help="Execute once; may call the explicitly configured model.")
    config_run.add_argument("path")
    config_input = config_run.add_mutually_exclusive_group(required=True)
    config_input.add_argument("--query", help="A plain query without cached task identity.")
    config_input.add_argument("--state", help="Visible state JSON including task_id, context and messages.")
    config_run.set_defaults(func=_cmd_config)

    prepare_adaptive = config_commands.add_parser("prepare-adaptive", help="Bind the controlled Adaptive-RAG router, generator and retriever; no inference or fitting.")
    prepare_adaptive.add_argument("path")
    prepare_adaptive.add_argument("--router-checkpoint", required=True)
    prepare_adaptive.add_argument("--output-dir", required=True)
    prepare_adaptive.set_defaults(func=_cmd_prepare_adaptive)

    prepare_qrag = config_commands.add_parser("prepare-qrag", help="Verify mounted Q-RAG assets and bind one visible state; no inference.")
    prepare_qrag.add_argument("path")
    prepare_qrag.add_argument("--state", required=True)
    prepare_qrag.add_argument("--reader-repository", required=True)
    prepare_qrag.add_argument("--reader-revision", required=True)
    prepare_qrag.add_argument("--reader-profile", required=True, choices=(
        "upstream_qwq32b_family_controlled", "factorial_reader_transfer",
        "factorial_instruction_preserving_reader_transfer",
        "factorial_native_query_reader_transfer"))
    prepare_qrag.add_argument("--output-dir", required=True)
    prepare_qrag.set_defaults(func=_cmd_prepare_qrag)

    prepare_longllmlingua = config_commands.add_parser("prepare-longllmlingua", help="Bind controlled compression to the explicit helper assets and reader; no compression or generation.")
    prepare_longllmlingua.add_argument("path")
    prepare_longllmlingua.add_argument("--output-dir", required=True)
    prepare_longllmlingua.set_defaults(func=_cmd_prepare_longllmlingua)

    prepare_recomp = config_commands.add_parser("prepare-recomp", help="Verify mounted RECOMP source/checkpoint and derive run configuration; no model calls or fitting.")
    prepare_recomp.add_argument("path")
    prepare_recomp.add_argument("--output-dir", required=True)
    prepare_recomp.set_defaults(func=_cmd_prepare_recomp)

    prepare_worker = config_commands.add_parser("prepare-worker", help="Bind installed process-worker files and verify mounted source; no inference or fitting.")
    prepare_worker.add_argument("path")
    prepare_worker.add_argument("--output-dir", required=True)
    prepare_worker.set_defaults(func=_cmd_prepare_worker)

    prepare_hyde = config_commands.add_parser("prepare-hyde", help="Bind controlled HyDE to the configured model and dense corpus; no generation or embedding.")
    prepare_hyde.add_argument("path")
    prepare_hyde.add_argument("--output-dir", required=True)
    prepare_hyde.set_defaults(func=_cmd_prepare_hyde)

    prepare_query2doc = config_commands.add_parser("prepare-query2doc", help="Bind controlled Query2doc to the configured model, retriever and native source; no model calls.")
    prepare_query2doc.add_argument("path")
    prepare_query2doc.add_argument("--output-dir", required=True)
    prepare_query2doc.set_defaults(func=_cmd_prepare_query2doc)

    prepare_ircot = config_commands.add_parser("prepare-ircot", help="Check native IRCoT assets and write a ready, explicitly substituted configuration; no model calls.")
    prepare_ircot.add_argument("path")
    prepare_ircot.add_argument("--output-dir", required=True)
    prepare_ircot.add_argument("--accept-substitutions", action="store_true")
    prepare_ircot.set_defaults(func=_cmd_prepare_ircot)

    config_fit = config_commands.add_parser('fit', help='Validate an explicit ACE/SkillOpt native fit; execute only on request.')
    config_fit.add_argument('path', help='Fit configuration JSON with explicit development data/assets and optimization controls.')
    config_fit.add_argument('--execute', action='store_true', help='Perform one native fit; may consume model calls, with no automatic retry.')
    config_fit.add_argument('--output-dir', help='New directory for attempt records and the reusable artifact.')
    config_fit.set_defaults(func=_cmd_config_fit)

    config_online = config_commands.add_parser('online', help='Validate an explicitly transductive ACE stream; execute only on request.')
    config_online.add_argument('path', help='Online JSON with explicit stream or pinned native feedback inputs and all budgets.')
    config_online.add_argument('--execute', action='store_true', help='Consume the stream once; includes adaptation, with no automatic retry.')
    config_online.add_argument('--output-dir', help='New directory for native stream, feedback and cost evidence.')
    config_online.set_defaults(func=_cmd_config_online)

    prepare_fit = config_commands.add_parser('prepare-fit', help='Build ACE native feedback and a ready fit configuration from existing captures; no model calls.')
    prepare_fit.add_argument('path', help='Preparation JSON with explicit disjoint development groups and a fit template.')
    prepare_fit.add_argument('--output-dir', required=True, help='New directory for prepared inputs and the ready fit file.')
    prepare_fit.set_defaults(func=_cmd_prepare_fit)

    config_evaluate = config_commands.add_parser("evaluate", help="Execute selected static cases through native scoring; never retry.")
    config_evaluate.add_argument("path")
    config_evaluate.add_argument("--benchmark", required=True)
    config_evaluate.add_argument("--split", required=True)
    config_cases = config_evaluate.add_mutually_exclusive_group(required=True)
    config_cases.add_argument("--case-id", action="append", help="Exact case ID; repeat to select more cases.")
    config_cases.add_argument("--limit", type=int, help="First N cases in stable dataset order; no random sample or full implicit run.")
    config_sources = config_evaluate.add_mutually_exclusive_group()
    config_sources.add_argument("--prepared-root", help="Existing verified production dataset root.")
    config_sources.add_argument("--cache-dir", help="Bounded automatic preparation/cache directory.")
    config_sources.add_argument("--beir-sample-config", help="Pinned BEIR sample root/manifest JSON; compatibility subset only, never a full-corpus score.")
    config_sources.add_argument("--trec-sample-config", help="Pinned TREC DL 2019/2020 sample root/manifest JSON; compatibility subset only, no download.")
    config_evaluate.add_argument("--offline", action="store_true")
    config_evaluate.add_argument("--accept-license-review", action="store_true")
    config_evaluate.add_argument("--predictor-config", help="Optional downstream predictor JSON (schema_version=1).")
    config_evaluate.add_argument("--qrag-reader-config", help="Q-RAG reader identity JSON; derive a fresh binding for every selected QA case.")
    config_evaluate.add_argument("--bfcl-source-archive", help="Reuse a pinned BFCL v1.3 archive without copying/downloading; otherwise cache-first automatic preparation. Required with --prepared-root.")
    config_evaluate.add_argument("--longbench-config", help="JSON paths to pinned LongBench data_zip, scorer_path and prompts_path; relative to this file, no download.")
    config_evaluate.add_argument("--output", required=True, help="New JSONL file; started and terminal records are flushed per case.")
    config_evaluate.set_defaults(func=_cmd_config_evaluate)

    arena = config_commands.add_parser("evaluate-arena", help="Run native arena cases in isolated processes with official scoring; never retry.")
    arena.add_argument("path", help="Run configuration: algorithm, model, seed and total budget.")
    arena.add_argument("--arena-config", required=True, help="Native checkout and arena/environment configuration.")
    selection = arena.add_mutually_exclusive_group(required=True)
    selection.add_argument("--case-id", action="append", help="Native case ID; repeat for additional cases.")
    selection.add_argument("--limit", type=int, help="Execute the first N captured native cases.")
    selection.add_argument("--list-cases", action="store_true", help="Capture/list native case IDs without model calls.")
    arena.add_argument("--capture-only", action="store_true",
                       help="Save selected initial states without agent evaluation; Tau user simulation incurs calls.")
    arena.add_argument("--output-dir", required=True, help="New directory for checkpoints, per-case receipts and native artifacts.")
    arena.set_defaults(func=_cmd_config_arena)

    prepare_arena = config_commands.add_parser("prepare-arena", help="Prepare an owned native arena environment and required scoring assets; no agent calls.")
    prepare_arena.add_argument("path", help="Native arena configuration JSON; scientific settings are preserved.")
    prepare_arena.add_argument("--environment", required=True, help="New bundle directory, or a completed matching bundle to verify/reuse.")
    prepare_arena.add_argument("--python", required=True, help="Explicit Python executable (STATE >=3.12; ToolSandbox/ToolTalk 3.10-3.12).")
    prepare_arena.add_argument("--allow-network", action="store_true", help="Allow pinned source, Python dependencies and missing scorer assets to be downloaded.")
    prepare_arena.add_argument("--toolkit-wheel", help="Use this exact toolkit wheel instead of downloading the caller's version from PyPI.")
    prepare_arena.set_defaults(func=_cmd_prepare_arena)

    data_parser = subparsers.add_parser(
        "datasets", help="Prepare source-locked datasets with bounded automatic downloads."
    )
    data_commands = data_parser.add_subparsers(dest="dataset_command", required=True)
    data_list = data_commands.add_parser("list", help="Discover datasets without downloading.")
    data_list.add_argument("--search", default="", help="Search ID, suite or task.")
    data_list.add_argument("--family", choices=[item.value for item in SemanticFamily])
    data_list.add_argument("--json", action="store_true")
    data_list.set_defaults(func=_cmd_dataset_list)
    data_show = data_commands.add_parser("show", help="Show source sizes and runtime prerequisites.")
    data_show.add_argument("cell_id")
    data_show.set_defaults(func=_cmd_dataset_show)
    data_prepare = data_commands.add_parser(
        "prepare", help="Reuse a cache or download missing pinned source files."
    )
    data_prepare.add_argument("cell_id")
    data_prepare.add_argument("--cache-dir", type=Path)
    data_prepare.add_argument("--offline", action="store_true")
    data_prepare.add_argument("--source-archive", type=Path, help="BFCL only: use an existing pinned source archive without copying it.")
    data_prepare.add_argument("--accept-license-review", action="store_true")
    data_prepare.add_argument(
        "--max-download-mib",
        type=int,
        default=512,
        help="Total missing source-file limit; default 512 MiB. No full multi-GB download by default.",
    )
    data_prepare.set_defaults(func=_cmd_dataset_prepare)
    data_prepare.add_argument(
        "--max-prepared-mib",
        type=int,
        default=512,
        help="Per-asset normalized-output limit; default 512 MiB. Keeps a further 1 GiB free.",
    )
    trec = data_commands.add_parser(
        "prepare-trec-sample", help="Prepare a bounded official-source TREC compatibility sample."
    )
    trec.add_argument("root", type=Path)
    trec.add_argument("--year", required=True, type=int, choices=(2019, 2020))
    trec.add_argument("--query-count", type=int, default=2)
    trec.add_argument("--query-offset", type=int, default=0)
    trec.add_argument("--offline", action="store_true")
    trec.add_argument("--accept-license-review", action="store_true")
    trec.add_argument("--max-download-mib", type=int, default=16)
    trec.add_argument("--corpus-revision")
    trec.add_argument("--source-sample-root", type=Path)
    trec.add_argument("--source-manifest-sha256")
    trec.set_defaults(func=_cmd_trec_sample)

    list_parser = subparsers.add_parser("list", help="List methods or release profiles.")
    list_parser.add_argument("--search", default="", help="Search method names and release IDs.")
    list_parser.add_argument(
        "--family", choices=[item.value for item in SemanticFamily], help="Filter by family."
    )
    list_parser.add_argument(
        "--tier", choices=[item.value for item in ReleaseTier], help="Filter by release tier."
    )
    list_parser.add_argument(
        "--status", choices=[item.value for item in SupportStatus], help="Filter by status."
    )
    list_parser.add_argument(
        "--delivery-tier",
        choices=["must_ship", "stretch", "blocked_or_post_v1"],
        help="Filter release profiles by engineering delivery tier.",
    )
    list_parser.add_argument(
        "--releases", action="store_true", help="List release profiles instead of methods."
    )
    list_parser.add_argument("--json", action="store_true", help="Emit JSON.")
    list_parser.set_defaults(func=_cmd_list)

    show_parser = subparsers.add_parser("show", help="Show one method or release profile.")
    show_parser.add_argument("identity", help="Canonical method ID or exact release ID.")
    show_parser.add_argument(
        "--parameters", action="store_true",
        help="Inspect an exact release's constructor parameters without instantiating it.",
    )
    show_parser.set_defaults(func=_cmd_show)

    doctor_parser = subparsers.add_parser("doctor", help="Run registry integrity diagnostics.")
    doctor_parser.add_argument("--source", help="Source registry to compare with the snapshot.")
    doctor_parser.add_argument(
        "--offline",
        action="store_true",
        help="Assert that diagnostics must not contact external services.",
    )
    doctor_parser.add_argument("--json", action="store_true", help="Emit JSON.")
    doctor_parser.set_defaults(func=_cmd_doctor)

    validate_parser = subparsers.add_parser(
        "validate-registry", help="Validate a source or packaged registry."
    )
    validate_parser.add_argument(
        "path", nargs="?", help="Registry path; defaults to checkout source."
    )
    validate_parser.add_argument(
        "--check-snapshot",
        action="store_true",
        help="Also require equality with the packaged snapshot.",
    )
    validate_parser.add_argument("--json", action="store_true", help="Emit JSON.")
    validate_parser.set_defaults(func=_cmd_validate_registry)

    smoke_parser = subparsers.add_parser(
        "smoke",
        help="Run the dependency-free Query2doc and BM25 reference loop.",
    )
    smoke_parser.add_argument("--json", action="store_true", help="Emit JSON.")
    smoke_parser.set_defaults(func=_cmd_smoke)

    selector_parser = subparsers.add_parser(
        "selector",
        help="Inspect selector catalog, visible features, and exact frozen artifacts.",
    )
    selector_subparsers = selector_parser.add_subparsers(
        dest="selector_command",
        required=True,
    )
    selector_list = selector_subparsers.add_parser("list", help="List selector cards.")
    selector_list.add_argument("--json", action="store_true", help="Emit JSON.")
    selector_list.set_defaults(func=_cmd_selector_list)

    selector_features = selector_subparsers.add_parser(
        "features",
        help="Extract evaluator-blind visible-state features.",
    )
    selector_features.add_argument("--query", default="")
    selector_features.add_argument("--instruction", default="")
    selector_features.add_argument("--tool-name", action="append", default=[])
    selector_features.set_defaults(func=_cmd_selector_features)

    selector_inspect = selector_subparsers.add_parser(
        "inspect",
        help="Inspect one exact content-addressed selector artifact.",
    )
    selector_inspect.add_argument("--store", type=Path, required=True)
    selector_inspect.add_argument("--selector-id", required=True)
    selector_inspect.add_argument("--revision", required=True)
    selector_inspect.add_argument("--sha256", required=True)
    selector_inspect.set_defaults(func=_cmd_selector_inspect)

    campaign_parser = subparsers.add_parser(
        "campaign",
        help=(
            "Prepare, diagnose, smoke, run, verify, and analyze the frozen existing-data campaign."
        ),
    )
    campaign_subparsers = campaign_parser.add_subparsers(
        dest="campaign_command",
        required=True,
    )

    campaign_doctor_parser = campaign_subparsers.add_parser(
        "doctor",
        help="Inspect real prepared data, method surfaces, and stateful-runtime blockers.",
    )
    campaign_doctor_parser.add_argument(
        "--prepared",
        type=Path,
        default=Path("data/normalized"),
    )
    campaign_doctor_parser.add_argument(
        "--readiness",
        type=Path,
        help="Adapter readiness matrix; defaults to the checkout catalog when present.",
    )
    campaign_doctor_parser.add_argument(
        "--target",
        choices=("static", "full"),
        default="full",
        help="Choose whether missing stateful runtime bridges are fatal.",
    )
    campaign_doctor_parser.set_defaults(func=_cmd_campaign_doctor)

    campaign_prepare_parser = campaign_subparsers.add_parser(
        "prepare",
        help="Acquire and preprocess accepted sources only; never touches D0 inventory cells.",
    )
    campaign_prepare_parser.add_argument(
        "--store",
        type=Path,
        default=Path("data/raw"),
    )
    campaign_prepare_parser.add_argument(
        "--prepared",
        type=Path,
        default=Path("data/normalized"),
    )
    campaign_prepare_parser.add_argument("--allow-network", action="store_true")
    campaign_prepare_parser.add_argument(
        "--accept-license-review",
        action="store_true",
        help="Process ClariQ locally only after reviewing its source terms.",
    )
    campaign_prepare_parser.set_defaults(func=_cmd_campaign_prepare)

    campaign_bootstrap_parser = campaign_subparsers.add_parser(
        "bootstrap",
        help=(
            "Verify or fetch exact public method/arena source checkouts; "
            "does not install packages, download weights, or run experiments."
        ),
    )
    campaign_bootstrap_parser.add_argument(
        "--root",
        type=Path,
        default=Path("operator/checkouts"),
    )
    campaign_bootstrap_parser.add_argument(
        "--method",
        action="append",
        help="Limit bootstrap to one method ID; repeat for multiple methods.",
    )
    campaign_bootstrap_parser.add_argument(
        "--arena",
        action="append",
        help="Prepare only these arena sources (repeatable); add --method to also select methods.",
    )
    campaign_bootstrap_parser.add_argument(
        "--no-arenas",
        action="store_true",
        help="Do not include official arena source checkouts; incompatible with --arena.",
    )
    campaign_bootstrap_parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Fetch absent or mismatched public checkouts at their exact pinned commits.",
    )
    campaign_bootstrap_parser.set_defaults(func=_cmd_campaign_bootstrap)

    campaign_smoke_parser = campaign_subparsers.add_parser(
        "smoke",
        help="Load and score one real prepared example from all seven static D2 cells.",
    )
    campaign_smoke_parser.add_argument(
        "--prepared",
        type=Path,
        default=Path("data/normalized"),
    )
    campaign_smoke_parser.set_defaults(func=_cmd_campaign_smoke)

    campaign_methods_parser = campaign_subparsers.add_parser(
        "methods",
        help="List exact launch commands and reviewed adapter gates for all 27 methods.",
    )
    campaign_methods_parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("configs/parity/methods.json"),
    )
    campaign_methods_parser.add_argument(
        "--readiness",
        type=Path,
        default=Path("configs/parity/adapter_readiness.json"),
    )
    campaign_methods_parser.set_defaults(func=_cmd_campaign_methods)

    campaign_run_parser = campaign_subparsers.add_parser(
        "run",
        help="Preflight a frozen manifest by default; execute only with --execute.",
    )
    campaign_run_parser.add_argument("manifest", type=Path)
    campaign_run_parser.add_argument("--driver")
    campaign_run_parser.add_argument("--no-resolve-units", action="store_true")
    campaign_run_parser.add_argument("--shard-count", type=int)
    campaign_run_parser.add_argument("--shard-index", type=int, default=0)
    campaign_run_parser.add_argument("--execute", action="store_true")
    campaign_run_parser.add_argument("--output-root", type=Path, default=Path("results"))
    campaign_run_parser.add_argument("--resume", action="store_true")
    campaign_run_parser.add_argument("--fail-fast", action="store_true")
    campaign_run_parser.set_defaults(func=_cmd_campaign_run)

    campaign_verify_parser = campaign_subparsers.add_parser(
        "verify",
        help="Verify manifest, plan, driver identity, and cell checksums.",
    )
    campaign_verify_parser.add_argument("run_dir", type=Path)
    campaign_verify_parser.add_argument("--allow-incomplete", action="store_true")
    campaign_verify_parser.set_defaults(func=_cmd_campaign_verify)

    campaign_analyze_parser = campaign_subparsers.add_parser(
        "analyze",
        help="Run the strict confirmatory analyzer over verified run directories.",
    )
    campaign_analyze_parser.add_argument("run_dir", type=Path, nargs="+")
    campaign_analyze_parser.add_argument("--protocol", default="auto")
    campaign_analyze_parser.add_argument("--metric", default="valid_success")
    campaign_analyze_parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    campaign_analyze_parser.add_argument("--permutation-samples", type=int, default=2_000)
    campaign_analyze_parser.add_argument("--random-seed", type=int, default=2027)
    campaign_analyze_parser.add_argument("--algorithm-failure-value", type=float)
    campaign_analyze_parser.set_defaults(func=_cmd_campaign_analyze)

    campaign_run_all_parser = campaign_subparsers.add_parser(
        "run-all",
        help=(
            "Preflight all 27 methods and ten accepted-source cells, then execute "
            "every frozen job with one explicit --execute."
        ),
    )
    campaign_run_all_parser.add_argument(
        "config",
        type=Path,
        help="Strict self-hashed one-click campaign bundle.",
    )
    campaign_run_all_parser.add_argument(
        "--execute",
        action="store_true",
        help="Start model/external-runtime calls only after every job preflight is ready.",
    )
    campaign_run_all_parser.set_defaults(func=_cmd_campaign_run_all)

    campaign_run_production_parser = campaign_subparsers.add_parser(
        "run-production",
        help=(
            "Preflight or start the complete ordered production DAG: data, "
            "27-method library parity/discovery, confirmations, selector, robustness, "
            "stateful replications, and paper export."
        ),
    )
    campaign_run_production_parser.add_argument(
        "config",
        type=Path,
        help="Strict self-hashed production campaign DAG.",
    )
    campaign_run_production_parser.add_argument(
        "--execute",
        action="store_true",
        help="Start the complete DAG only after every initial prerequisite passes.",
    )
    campaign_run_production_parser.set_defaults(func=_cmd_campaign_run_production)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except AgentInputLabError as exc:
        if args.debug:
            raise
        print(f"agentinputlab: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
