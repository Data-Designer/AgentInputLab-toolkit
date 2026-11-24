"""User-facing static benchmark execution, reusing the native adapters/scorers.

No inferred labels, automatic retries, training, or stateful-arena substitution.
Model execution sees only a FrozenAgentState; gold stays with the scorer.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from agentinputlab.configuration import RunConfig
from agentinputlab.contracts import ExecutionBudget, InterventionContext, PatchOp
from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import to_jsonable
from agentinputlab.observability import BudgetLedger, RunMonitor
from agentinputlab.observability.instrumentation import ObservedLLMBackend, ObservedRetriever


class _AwaitingClarification(Exception):
    """Internal control flow: a static case cannot invent a user response."""

    def __init__(self, questions: list[str]) -> None:
        self.questions = questions


class _CapturedBackend:
    """Keep usable responses even when downstream parsing/scoring fails."""

    def __init__(self, inner: Any, *, capture_tool_calls: bool = False) -> None:
        self.inner = inner
        self.capture_tool_calls = capture_tool_calls
        self.calls: list[dict[str, Any]] = []

    @property
    def model_id(self):
        return self.inner.model_id

    @property
    def capabilities(self):
        return self.inner.capabilities

    def fingerprint(self):
        return self.inner.fingerprint()

    @property
    def complete_native(self):
        if not callable(getattr(self.inner, "complete_native", None)):
            raise AttributeError("wrapped backend does not implement native completions")
        return self._complete_native

    def _complete_native(self, parameters, *, seed):
        """Retain native text/logprobs and usage without changing provider parameters."""
        from agentinputlab.backends.llm.decomp_completion import validate_native_parameters

        parameters = validate_native_parameters(parameters)
        entry = {"request": to_jsonable(parameters), "seed": seed,
                 "transport": "native_completion", "status": "started"}
        self.calls.append(entry)
        try:
            response = self.inner.complete_native(parameters, seed=seed)
        except Exception as exc:
            entry.update(status="failed", error_type=type(exc).__name__)
            raise
        if isinstance(response, Mapping):
            entry.update(status="responded", response=to_jsonable({
                key: response[key] for key in ("model", "choices", "usage") if key in response}))
        else:
            entry.update(status="invalid_response", response_type=type(response).__name__)
        return response

    def generate(self, request):
        from agentinputlab.backends.llm import GenerationResponse

        entry = {"request": to_jsonable(request), "status": "started"}
        self.calls.append(entry)
        try:
            response = self.inner.generate(request)
        except Exception as exc:
            entry.update(status="failed", error_type=type(exc).__name__)
            raise
        if isinstance(response, GenerationResponse):
            # Provider raw envelopes/headers are unnecessary for exact text reuse.
            entry.update(status="responded", response={
                "text": response.text, "model": response.model, "finish_reason": response.finish_reason,
                "usage": to_jsonable(response.usage)})
            if self.capture_tool_calls:
                try:
                    calls = response.raw["choices"][0]["message"].get("tool_calls")
                    entry["response"]["tool_calls"] = to_jsonable(calls)
                except (KeyError, IndexError, TypeError, AttributeError):
                    pass  # Preserve text/usage even when the native envelope is invalid.
        else:
            entry.update(status="invalid_response", response_type=type(response).__name__)
        return response


def _remaining(budget: ExecutionBudget, ledger: BudgetLedger) -> ExecutionBudget:
    ledger.require_within_budget()
    usage = ledger.usage
    return ExecutionBudget(
        max_model_calls=budget.max_model_calls - usage.model_calls,
        max_retrieval_calls=budget.max_retrieval_calls - usage.retrieval_calls,
        max_tool_calls=budget.max_tool_calls - usage.tool_calls,
        max_output_tokens=budget.max_output_tokens - usage.output_tokens,
        max_context_tokens=(None if budget.max_context_tokens is None
                            else budget.max_context_tokens - usage.input_tokens),
        max_cost_usd=(None if budget.max_cost_usd is None else budget.max_cost_usd - usage.cost_usd),
    )


def evaluate_config(
    config: RunConfig,
    benchmark_id: str,
    *,
    split: str,
    example_ids: Sequence[str] | None = None,
    limit: int | None = None,
    prepared_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
    offline: bool = False,
    accept_license_review: bool = False,
    predictor_parameters: Mapping[str, Any] | None = None,
    context: InterventionContext | None = None,
    adapter: Any = None,
    on_record: Callable[[Mapping[str, Any]], None] | None = None,
    qrag_reader: Mapping[str, Any] | None = None,
    bfcl_source_archive: str | Path | None = None,
    longbench_sources: Mapping[str, Any] | None = None,
    beir_sample_sources: Mapping[str, Any] | None = None,
    trec_sample_sources: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run explicitly selected cases once; report failures without fabricated scores.

    The default loader verifies prepared artifacts or prepares a bounded cache.
    Python callers may supply a native adapter. CLI callers use verified loaders.
    ``on_record`` receives a started marker before execution and a terminal row
    afterwards, allowing a caller to persist evidence even if interrupted.
    """
    from agentinputlab.benchmarks import load_production_benchmark_adapter, prepare_benchmark
    from agentinputlab.benchmarks.runtime import build_benchmark_example_record
    from agentinputlab.experiments.drivers.static_production import (
        StaticPredictorBuildRequest,
        StaticRuntimeComponents,
    )
    from agentinputlab.experiments.production_builders import (
        ProductionPredictorBuilder,
        _parse_visible_qa_prediction,
    )
    from agentinputlab.runner import CompositeInterventionRunner, InterventionRunner

    if example_ids is not None and (isinstance(example_ids, (str, bytes)) or not isinstance(example_ids, Sequence)
            or not example_ids or any(not isinstance(x, str) or not x.strip() for x in example_ids)
            or len(set(example_ids)) != len(example_ids)):
        raise ValidationError("example_ids must be a non-empty sequence of unique case IDs")
    if ((example_ids is None) == (limit is None)
            or (limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1))):
        raise ValidationError("choose unique example_ids or a positive limit, not both")
    if not isinstance(offline, bool) or not isinstance(accept_license_review, bool):
        raise ValidationError("offline and accept_license_review must be booleans")
    if on_record is not None and not callable(on_record):
        raise ValidationError("on_record must be callable")
    if adapter is not None and (prepared_root is not None or cache_dir is not None):
        raise ValidationError("choose an adapter or dataset paths, not both")
    if prepared_root is not None and cache_dir is not None:
        raise ValidationError("choose prepared_root or cache_dir, not both")
    if trec_sample_sources is not None:
        if (adapter is not None or prepared_root is not None or cache_dir is not None
                or bfcl_source_archive is not None or longbench_sources is not None
                or beir_sample_sources is not None or config.retriever.get("kind") == "beir_archive"):
            raise ValidationError("TREC sample owns its dataset source; do not also supply other dataset sources")
        from agentinputlab.trec_sample_configuration import load

        adapter = load(trec_sample_sources, benchmark_id=benchmark_id, split=split,
                       accept_license_review=accept_license_review)
    if beir_sample_sources is not None:
        if (adapter is not None or prepared_root is not None or cache_dir is not None
                or bfcl_source_archive is not None or longbench_sources is not None
                or config.retriever.get("kind") == "beir_archive"):
            raise ValidationError("BEIR sample owns its dataset source; do not also supply other dataset sources")
        from agentinputlab.beir_sample_configuration import load

        adapter = load(beir_sample_sources, benchmark_id=benchmark_id, split=split,
                       accept_license_review=accept_license_review)
    if (config.retriever.get("kind") in {"prepared_corpus", "beir_archive"}
            and config.retriever.get("benchmark_id") != benchmark_id):
        raise ValidationError("retriever benchmark must match the evaluated benchmark")
    if (config.retriever.get("kind") == "sqlite_fts5" and "benchmark_id" in config.retriever
            and config.retriever["benchmark_id"] != benchmark_id):
        raise ValidationError("retriever benchmark must match the evaluated benchmark")
    if config.retriever.get("kind") == "beir_archive" and (
        adapter is not None or prepared_root is not None or cache_dir is not None
    ):
        raise ValidationError("BEIR retriever configuration owns its dataset source; do not also supply adapter or dataset paths")
    if config.retriever.get("kind") == "beir_archive":
        configured_split = config.retriever.get("split", split)
        if configured_split != split:
            raise ValidationError("BEIR retriever split must match the evaluated split")
        if split != "test":
            config = replace(config, retriever={**config.retriever, "split": split})
    if (config.vector_runtime.get("kind") == "prepared_corpus"
            and config.vector_runtime.get("benchmark_id") != benchmark_id):
        raise ValidationError("vector runtime benchmark must match the evaluated benchmark")

    qrag_binder = None
    if qrag_reader is not None:
        from agentinputlab.qrag_configuration import QRAGBatchBinder

        qrag_binder = QRAGBatchBinder(config, qrag_reader)
        config = qrag_binder.base
    if offline and config.retriever.get("kind") == "beir_archive":
        config = replace(config, retriever={**config.retriever, "offline": True})
    base_context = config._configured_context(context)
    if qrag_binder is not None and qrag_binder.candidate_source and base_context.retriever is None:
        raise ValidationError("Q-RAG candidate_source requires a configured corpus retriever")
    if config.retriever.get("kind") == "beir_archive":
        from agentinputlab.beir_configuration import BEIRCorpusRetriever

        if not isinstance(base_context.retriever, BEIRCorpusRetriever):
            raise ValidationError("BEIR file configuration cannot be overridden by a different context retriever")
        adapter = base_context.retriever.adapter
    is_bfcl = benchmark_id == "bfcl.v3.simple_python"
    from agentinputlab.benchmarks.longbench import _SUBSETS as longbench_subsets

    is_longbench = benchmark_id in longbench_subsets
    if qrag_binder is not None and qrag_binder.longbench_context and not is_longbench:
        raise ValidationError("Q-RAG longbench_context applies only to LongBench")
    if longbench_sources is not None and not is_longbench:
        raise ValidationError("longbench_sources applies only to LongBench cells")
    if bfcl_source_archive is not None and not is_bfcl:
        raise ValidationError("bfcl_source_archive applies only to bfcl.v3.simple_python")
    method = config.create_intervention()
    is_patch = callable(getattr(method, "propose", None))
    parameters = {"schema_version": 1} if predictor_parameters is None else predictor_parameters
    trace_ranking = False
    if isinstance(parameters, Mapping) and "trace_ranking" in parameters:
        from agentinputlab.benchmarks.trace_retrieval import validate_trace_ranking

        parameters = dict(parameters)
        trace_options = parameters.pop("trace_ranking")
        validate_trace_ranking(trace_options, parameters, benchmark_id=benchmark_id,
                               release_id=config.release_id, has_retriever=base_context.retriever is not None)
        if not is_patch or qrag_binder is not None:
            raise ValidationError("trace_ranking requires a patch profile without an additional reader")
        trace_ranking = True
    ircot_input = None
    if isinstance(parameters, Mapping) and "ircot_input" in parameters:
        from agentinputlab.ircot_configuration import validate_evaluation_input

        parameters = dict(parameters)
        ircot_input = parameters.pop("ircot_input")
        validate_evaluation_input(ircot_input, config, benchmark_id=benchmark_id, split=split)
    archived_retrieval = None
    if isinstance(parameters, Mapping) and "archived_retrieval" in parameters:
        from agentinputlab.archived_retrieval_configuration import load as load_archived_retrieval

        parameters = dict(parameters)
        archived_retrieval = load_archived_retrieval(parameters.pop("archived_retrieval"), method,
            benchmark_id=benchmark_id, split=split)
    archived_input = None
    if isinstance(parameters, Mapping) and "archived_input" in parameters:
        parameters = dict(parameters)
        archived_input = parameters.pop("archived_input")
        # Null must not silently disable a requested projection.
        if archived_input is None:
            raise ValidationError("archived_input must explicitly select a projection")
    retriever_capability = None
    if isinstance(parameters, Mapping) and "retriever_capability" in parameters:
        parameters = dict(parameters)
        retriever_capability = parameters.pop("retriever_capability")
        if (not isinstance(retriever_capability, Mapping)
                or dict(retriever_capability) != {"visible_name": "search"}
                or config.release_id != "plan.self_ask.full_loop"
                or base_context.retriever is None):
            raise ValidationError("retriever_capability requires visible_name=search, "
                                  "plan.self_ask.full_loop, and a configured retriever")
    history_options = None
    if isinstance(parameters, Mapping) and "history" in parameters:
        parameters = dict(parameters)
        history_options = parameters.pop("history")
        if (not isinstance(history_options, Mapping) or dict(history_options) != {"kind": "acon"}
                or config.release_id != "curate.acon.history_runtime"):
            raise ValidationError("history requires kind=acon and curate.acon.history_runtime")
    composite_retrieval = None
    context_prefill = None
    patch_retrieval = None
    retrieval_prefill = None
    if isinstance(parameters, Mapping) and "retrieval_prefill" in parameters:
        from agentinputlab.retrieval_prefill_configuration import options as prefill_options

        parameters = dict(parameters)
        retrieval_prefill = prefill_options(parameters.pop("retrieval_prefill"))
        if ("patch_retrieval" in parameters or "composite_retrieval" in parameters
                or not is_patch or base_context.retriever is None or qrag_binder is not None
                or config.retriever.get("kind") == "visible_context_bm25"):
            raise ValidationError("retrieval_prefill requires a patch profile and configured corpus retriever; "
                                  "do not combine it with patch_retrieval or composite_retrieval")
        context_prefill = retrieval_prefill
    if isinstance(parameters, Mapping) and "patch_retrieval" in parameters:
        from agentinputlab.patch_retrieval_configuration import options as patch_options

        parameters = dict(parameters)
        patch_retrieval = patch_options(parameters.pop("patch_retrieval"))
        if "composite_retrieval" in parameters:
            raise ValidationError("patch_retrieval and composite_retrieval are mutually exclusive")
        context_prefill = patch_retrieval.get("context_prefill")
        if context_prefill is not None and (
                base_context.retriever is None or qrag_binder is not None
                or config.retriever.get("kind") == "visible_context_bm25"):
            raise ValidationError("patch_retrieval context_prefill requires a configured corpus retriever")
    if isinstance(parameters, Mapping) and "composite_retrieval" in parameters:
        parameters = dict(parameters)
        composite_retrieval = parameters.pop("composite_retrieval")
        if (not isinstance(composite_retrieval, Mapping)
                or set(composite_retrieval) - {"output_format", "allow_json_fence", "context_prefill"}
                or composite_retrieval.get("output_format") != "json_query"
                or type(composite_retrieval.get("allow_json_fence", False)) is not bool):
            raise ValidationError("composite_retrieval requires output_format=json_query and an optional boolean allow_json_fence")
        if "context_prefill" in composite_retrieval:
            from agentinputlab.retrieval_prefill_configuration import options as prefill_options

            context_prefill = prefill_options(composite_retrieval["context_prefill"])
            if (config.release_id not in {"plan.plan_and_budget.full", "plan.plan_and_budget.hard_cap_extension"}
                    or base_context.retriever is None or qrag_binder is not None
                    or config.retriever.get("kind") == "visible_context_bm25"):
                raise ValidationError("context_prefill requires Plan-and-Budget full/hard-cap and a configured corpus retriever")
    if archived_input is not None:
        from agentinputlab.ircot_replay_configuration import validate as validate_archived_input

        validate_archived_input(archived_input, method, benchmark_id=benchmark_id, split=split)
        if any(x is not None for x in (context_prefill, qrag_binder, retriever_capability, history_options)):
            raise ValidationError("archived_input cannot be combined with other input preparation")
    if ircot_input is not None:
        from agentinputlab.experiments.production_builders import _parse_predictor_parameters

        if any(x is not None for x in (archived_input, archived_retrieval, context_prefill,
                qrag_binder, retriever_capability, history_options, composite_retrieval, patch_retrieval)):
            raise ValidationError("ircot_input cannot be combined with other input preparation")
        if _parse_predictor_parameters(parameters)["qa"]["output_format"] != "answer_only":
            raise ValidationError("ircot_input requires qa.output_format=answer_only")
    runtime = StaticRuntimeComponents(backend=base_context.backend, retriever=base_context.retriever,
                                      score_only=not is_patch)
    request = StaticPredictorBuildRequest(
        benchmark_id=benchmark_id, split=split, model_slot="configured", seed=config.seed,
        builder_parameters=parameters, runtime=runtime,
    )
    builder = ProductionPredictorBuilder()
    # Reject unsupported datasets/predictor settings before download or model use.
    if is_bfcl:
        from agentinputlab import bfcl_configuration

        bfcl_settings = bfcl_configuration.options(parameters, split=split, backend=base_context.backend)
        if qrag_reader is not None or base_context.tools is not None:
            raise ValidationError("BFCL binds its own native tool catalog; do not supply QA reader or custom tools")
        if adapter is None and prepared_root is not None and bfcl_source_archive is None:
            raise ValidationError("BFCL --prepared-root requires --bfcl-source-archive; alternatively use --cache-dir for automatic preparation")
        predictor = None
    elif is_longbench:
        from agentinputlab import longbench_configuration

        longbench_settings = longbench_configuration.options(parameters, split=split, backend=base_context.backend,
                                                             prediction_required=is_patch)
        if longbench_settings.get("rewrite_retrieval_top_k") is not None and (
                benchmark_id not in longbench_configuration.LONG_TEXT_REWRITE_CELLS
                or config.release_id not in longbench_configuration.LONG_TEXT_REWRITE_RELEASES
                or config.retriever.get("kind") != "visible_context_bm25"
                or "chunk_words" not in config.retriever):
            raise ValidationError("LongBench rewrite retrieval requires a supported rewriter and a chunked visible corpus")
        if longbench_settings.get("preserve_original_question") and qrag_binder is not None:
            raise ValidationError("LongBench native-question projection does not alter an explicit Q-RAG reader")
        if longbench_settings.get("reformer_context_top_k") is not None and (
                config.release_id != "rewrite.reformer.repo_native_r_only"
                or longbench_settings["reformer_context_top_k"] != getattr(method, "top_document_count", None)):
            raise ValidationError("ReFormeR prefill must match the native pattern selector's document count")
        if prepared_root is not None:
            raise ValidationError("LongBench uses longbench_sources or cache_dir rather than prepared_root")
        if longbench_sources is not None and cache_dir is not None:
            raise ValidationError("choose LongBench longbench_sources or cache_dir, not both")
        if adapter is not None and longbench_sources is not None:
            raise ValidationError("choose a LongBench adapter or longbench_sources, not both")
        if adapter is None:
            if longbench_sources is None:
                from agentinputlab.benchmarks import load_benchmark

                adapter = load_benchmark(benchmark_id, cache_dir=cache_dir, offline=offline,
                    accept_license_review=accept_license_review)
            else:
                from agentinputlab.benchmarks.longbench import LongBenchAdapter

                adapter = LongBenchAdapter(benchmark_id=benchmark_id,
                    **longbench_configuration.sources(longbench_sources))
        predictor = None
    else:
        predictor = builder.build(request, prediction_required=(is_patch and not trace_ranking) or composite_retrieval is not None)
    if retriever_capability is not None:
        if is_bfcl or (not is_longbench and not trace_ranking and predictor.fingerprint()["task"] != "qa"):
            raise ValidationError("retriever_capability requires static QA, LongBench, or observed ranking")
        if base_context.tools is not None:
            raise ValidationError("retriever_capability cannot replace a native tool executor")
    if history_options is not None:
        if is_bfcl or is_longbench or predictor.fingerprint()["task"] != "qa":
            raise ValidationError("static history currently requires the QA reader; native arenas use arena history")
        from agentinputlab.experiments.production_builders import _parse_predictor_parameters

        history_qa = _parse_predictor_parameters(parameters)["qa"]
        if not history_qa["system_prompt"]:
            raise ValidationError("static history requires the reader's existing nonempty system prompt")
    qrag_qa_preserve_empty_answer = False
    if config.runtime.get("kind") == "qrag_native":
        from agentinputlab.algorithms.acquire.qrag.reader_adapter import (
            QRAG_INSTRUCTION_READER_PROFILE,
            QRAG_QUERY_READER_PROFILE,
        )

        reader_profile = method.artifacts.reader_profile
        # The explicit instruction reader requires a string, including
        # an empty answer that the original QA scorer can grade.
        qrag_qa_preserve_empty_answer = reader_profile == QRAG_INSTRUCTION_READER_PROFILE
        if composite_retrieval is not None and reader_profile == QRAG_INSTRUCTION_READER_PROFILE:
            raise ValidationError("Q-RAG QA reader conflicts with query-only retrieval; explicitly select "
                                  "factorial_native_query_reader_transfer and prepare its identity")
        if reader_profile == QRAG_QUERY_READER_PROFILE and composite_retrieval is None:
            raise ValidationError("Q-RAG query reader requires composite_retrieval output_format=json_query")
    if is_longbench and is_patch and longbench_settings.get("composite_output_format", "text") != "text":
        raise ValidationError("LongBench composite_output_format applies only to answer-producing profiles")
    if not is_patch and not callable(getattr(method, "run", None)):
        raise ValidationError("this recipe requires fit/apply before benchmark evaluation")
    if retrieval_prefill is not None:
        from agentinputlab.experiments.production_builders import _parse_predictor_parameters

        retrieval_options = _parse_predictor_parameters(parameters)["retrieval"]
        if (is_bfcl or is_longbench or predictor.fingerprint()["task"] != "retrieval"
                or retrieval_options["query_mode"] != "query_only"
                or retrieval_options["source_mode"] != "retriever"):
            raise ValidationError("retrieval_prefill requires retrieval evaluation with query_only "
                                  "and retriever source_mode; it does not add a model reader")
    if patch_retrieval is not None:
        from agentinputlab.experiments.production_builders import _parse_predictor_parameters

        retrieval_options = _parse_predictor_parameters(parameters)["retrieval"]
        if (not is_patch or is_bfcl or is_longbench or predictor.fingerprint()["task"] != "retrieval"
                or retrieval_options["query_mode"] != "query_only"
                or retrieval_options["source_mode"] != "retriever"
                or base_context.backend is None or base_context.retriever is None):
            raise ValidationError("patch_retrieval requires a patch retrieval profile, backend, corpus "
                                  "retriever, query_only and retriever source_mode")
    if composite_retrieval is not None:
        from agentinputlab.experiments.production_builders import _parse_predictor_parameters

        retrieval_options = _parse_predictor_parameters(parameters)["retrieval"]
        if (is_patch or is_bfcl or is_longbench or predictor.fingerprint()["task"] != "retrieval"
                or retrieval_options["query_mode"] != "query_only"
                or retrieval_options["source_mode"] != "retriever"):
            raise ValidationError("composite_retrieval requires a composite retrieval profile with query_only and retriever source_mode")
    if archived_retrieval is not None and (
            is_bfcl or is_longbench or predictor.fingerprint()["task"] != "retrieval"
            or any(v is not None for v in (composite_retrieval, patch_retrieval, context_prefill,
                                          history_options, retriever_capability, qrag_binder))
            or base_context.backend is not None or base_context.retriever is not None):
        raise ValidationError("archived_retrieval is score-only; do not combine it with live models, retrieval or preparation")
    if composite_retrieval is None and archived_retrieval is None and not is_patch and (is_bfcl or (not is_longbench and predictor.fingerprint()["task"] != "qa")):
        raise ValidationError("answer-producing profiles require a QA prediction/scorer binding")
    if adapter is None:
        from agentinputlab.benchmarks.beir_archive import BEIR_ARCHIVE_LOCKS

        if benchmark_id in BEIR_ARCHIVE_LOCKS:
            from agentinputlab.benchmarks import load_benchmark

            if prepared_root is not None:
                raise ValidationError("BEIR archive cells use cache_dir or retriever.source_archive, not prepared_root")
            adapter = load_benchmark(benchmark_id, cache_dir=cache_dir, offline=offline,
                                     accept_license_review=accept_license_review, split=split)
    if adapter is None:
        if prepared_root is None:
            prepared = prepare_benchmark(benchmark_id, cache_dir=cache_dir, offline=offline,
                                         accept_license_review=accept_license_review,
                                         **({"source_archive": bfcl_source_archive} if is_bfcl else {}))
            prepared_root = prepared["prepared_root"]
            if is_bfcl:
                bfcl_source_archive = prepared["source_archive"]
        if is_bfcl:
            from agentinputlab.benchmarks import (
                SourceVerifiedBFCLV3ASTBridge,
                load_bfcl_v3_simple_adapter,
            )

            bridge = SourceVerifiedBFCLV3ASTBridge(source_archive=bfcl_source_archive,
                                                  model_id=base_context.backend.model_id)
            adapter = load_bfcl_v3_simple_adapter(prepared_root, evaluation_bridge=bridge)
        else:
            adapter = load_production_benchmark_adapter(benchmark_id, prepared_root)
    if (adapter.benchmark_id != benchmark_id
            or getattr(adapter, "pairing_mode", None) != "stateless_score_twice"):
        raise ValidationError("benchmark identity mismatch or native stateful continuation required")
    inventory = {}
    for example in adapter.examples(split):
        if example.example_id in inventory:
            raise ValidationError("adapter contains duplicate example IDs")
        inventory[example.example_id] = example
    if example_ids is None:
        example_ids = tuple(inventory)[:limit]
    if not example_ids:
        raise ValidationError("selected dataset split is empty")
    missing = set(example_ids) - inventory.keys()
    if missing:
        raise ValidationError("unknown example IDs: " + ", ".join(sorted(missing)))

    sample_metadata = ({"dataset_fingerprint": to_jsonable(adapter.fingerprint())}
                       if trec_sample_sources is not None or beir_sample_sources is not None else {})
    records = []
    for example_id in example_ids:
        example = inventory[example_id]
        state = example.visible_state(budget=config.budget)
        if is_bfcl:
            state, bfcl_functions, bfcl_tools = bfcl_configuration.visible_state(adapter, example, config.budget)
        ledger, monitor = BudgetLedger(config.budget), RunMonitor()
        captured = (_CapturedBackend(base_context.backend, capture_tool_calls=is_bfcl)
                    if base_context.backend is not None else None)
        case_context = replace(base_context, backend=captured, budget_ledger=ledger, monitor=monitor)
        if is_bfcl:
            case_context = replace(case_context, tools=bfcl_tools)
        row: dict[str, Any] = {"example_id": example_id, "input_state_hash": state.state_hash,
                               "status": "started", "scored": False, **sample_metadata}
        if on_record is not None:
            on_record(to_jsonable(row))
        stage = "intervention"
        history_hook = None
        retrieval_evidence = None
        reformer_original = None
        try:
            if ircot_input is not None:
                from agentinputlab.ircot_configuration import prepare_evaluation_input

                stage = "ircot_input"
                state, preparation = prepare_evaluation_input(state, config, case_context.retriever)
                row["ircot_input"] = preparation
                row["source_input_state_hash"] = row["input_state_hash"]
                row["input_state_hash"] = state.state_hash
                stage = "intervention"
            if archived_input is not None:
                from agentinputlab.ircot_replay_configuration import (
                    prepare as prepare_archived_input,
                )

                stage = "archived_input"
                state, preparation = prepare_archived_input(state, method, options=archived_input, adapter=adapter)
                row["archived_input"] = preparation
                row["source_input_state_hash"] = row["input_state_hash"]
                row["input_state_hash"] = state.state_hash
                stage = "intervention"
            if context_prefill is not None:
                from agentinputlab.retrieval_prefill_configuration import prepare as prepare_prefill

                stage = "context_prefill"
                with monitor.run_scope(release_id=config.release_id + ".context_prefill",
                                       state_hash=state.state_hash) as run_id:
                    observed = ObservedRetriever(case_context.retriever, monitor, ledger, run_id)
                    state, preparation = prepare_prefill(state, observed, context_prefill)
                state = replace(state, budget=_remaining(config.budget, ledger))
                preparation["execution_state_sha256"] = state.state_hash
                row["context_prefill"] = preparation
                row["source_input_state_hash"] = row["input_state_hash"]
                row["input_state_hash"] = state.state_hash
                stage = "intervention"
            if qrag_binder is not None and qrag_binder.candidate_source:
                stage = "candidate_retrieval"
                with monitor.run_scope(release_id=config.release_id + ".candidates",
                                       state_hash=state.state_hash) as run_id:
                    observed = ObservedRetriever(case_context.retriever, monitor, ledger, run_id)
                    state, preparation = qrag_binder.prepare_candidates(state, observed)
                state = replace(state, budget=_remaining(config.budget, ledger))
                preparation["execution_state_sha256"] = state.state_hash
                row["candidate_preparation"] = preparation
                row["source_input_state_hash"] = row["input_state_hash"]
                row["input_state_hash"] = state.state_hash
                stage = "intervention"
            if qrag_binder is not None and qrag_binder.longbench_context:
                stage = "longbench_context_preparation"
                with monitor.run_scope(release_id=config.release_id + ".longbench_context",
                                       state_hash=state.state_hash) as run_id:
                    state, preparation = longbench_configuration.prepare_qrag_context(
                        state, adapter=adapter,
                        observe_retriever=lambda r, m=monitor, ledger=ledger, rid=run_id: ObservedRetriever(
                            r, m, ledger, rid))
                row["candidate_preparation"] = preparation
                row["source_input_state_hash"] = row["input_state_hash"]
                row["input_state_hash"] = state.state_hash
                stage = "intervention"
            case_config = (qrag_binder.bind(state, budget=state.budget)
                           if qrag_binder is not None else config)
            if context_prefill is not None:
                case_config = replace(case_config, budget=state.budget)
            if qrag_binder is not None:
                from agentinputlab.contracts.serialization import content_hash

                row["qrag_binding"] = {"artifact_identity_sha256": content_hash(case_config.parameters["artifacts"]),
                                       "state_sha256": state.state_hash,
                                       "mode": ("longbench_visible_projection" if qrag_binder.longbench_context
                                                else "retrieved_visible_candidates" if qrag_binder.candidate_source
                                                else "given_visible_context")}
            case_context = case_config._bind_case_context(case_context, state)
            if is_longbench and longbench_settings.get("reformer_context_top_k") is not None:
                reformer_original = state
                stage = "reformer_visible_prefill"
                with monitor.run_scope(release_id=config.release_id + ".visible_prefill", state_hash=state.state_hash) as run_id:
                    observed = ObservedRetriever(case_context.retriever, monitor, ledger, run_id)
                    state, preparation = longbench_configuration.prepare_reformer_documents(
                        state, observed, top_k=longbench_settings["reformer_context_top_k"])
                state = replace(state, budget=_remaining(config.budget, ledger))
                case_config = replace(case_config, budget=state.budget)
                row["reformer_visible_prefill"] = {**preparation, "execution_state_sha256": state.state_hash}
                row["source_input_state_hash"] = row["input_state_hash"]
                row["input_state_hash"] = state.state_hash
                stage = "intervention"
            if retriever_capability is not None:
                # Declare the actual Self-Ask search dependency, not an invented
                # tool or a benchmark/gold override. Binding must succeed first.
                if state.tool_names not in ((), ("search",)):
                    raise ValidationError("retriever_capability conflicts with existing visible tools")
                source_hash = state.state_hash
                state = replace(state, tool_names=("search",))
                row["source_input_state_hash"] = source_hash
                row["input_state_hash"] = state.state_hash
                row["retriever_capability"] = {
                    "visible_name": "search", "binding": "self_ask_configured_retriever.v1",
                    "fingerprint": dict(case_context.retriever.fingerprint()),
                    "native_tool_executor_added": False,
                }
            if config.retriever.get("kind") == "visible_context_bm25":
                row["case_retriever"] = {
                    "binding": "visible_context_bm25.v1",
                    "input_state_hash": state.state_hash,
                    "fingerprint": dict(case_context.retriever.fingerprint()),
                    "full_corpus_retrieval": False,
                }
            if config.runtime.get("kind") == "querygym_query2doc":
                # The native runtime calls its bound backend directly. Bind the
                # case capture proxy (not a second budget wrapper) so both its
                # response and the reader response are retained, charged once.
                case_context = replace(case_context, external_runtime=config._runtime(
                    load=True, backend=captured))
            if trace_ranking:
                from agentinputlab.benchmarks.trace_retrieval import CaseRetrievalEvidence

                retrieval_evidence = CaseRetrievalEvidence(case_context.retriever)
                case_context = replace(case_context, retriever=retrieval_evidence)
            # A fresh method avoids carrying fitted/mutable state between cases.
            method = case_config.create_intervention()
            # Runners require a pristine ledger. Candidate retrieval is a
            # distinct budgeted stage; hand over only its remainder, then merge
            # actual usage even on failure before downstream work can proceed.
            execution_ledger = (BudgetLedger(case_config.budget)
                                if context_prefill is not None or reformer_original is not None or (
                                    qrag_binder is not None and (qrag_binder.candidate_source
                                                                or qrag_binder.longbench_context)) else None)
            if execution_ledger is not None:
                case_context = replace(case_context, budget_ledger=execution_ledger)
            try:
                if history_options is not None:
                    from agentinputlab.integrations.live_history import ACONLiveHistory

                    history_hook = ACONLiveHistory(method,
                        replace(case_context, budget_ledger=None, monitor=None),
                        per_call_output_tokens=history_qa["max_tokens"])
                    intervention = None
                    transformed = state
                    row["execution_mode"] = "static_QA_native_history_boundary"
                elif is_patch:
                    intervention = InterventionRunner().run(method, state, case_context)
                    transformed = intervention.patch.apply(state)
                else:
                    intervention = CompositeInterventionRunner().run(method, state, case_context)
                    transformed = state
            finally:
                if execution_ledger is not None:
                    ledger.reserve(execution_ledger.usage)
            row["intervention"] = to_jsonable(intervention)
            if is_patch and intervention is not None:
                questions = [str(op.value) for op in intervention.patch.operations
                             if op.op is PatchOp.REQUEST_CLARIFICATION]
                if questions:
                    ledger.require_within_budget()
                    raise _AwaitingClarification(questions)
            stage = "downstream"
            with monitor.run_scope(release_id=config.release_id + ".downstream", state_hash=transformed.state_hash) as run_id:
                if trace_ranking:
                    from agentinputlab.benchmarks.trace_retrieval import prediction_from_trace

                    prediction = prediction_from_trace(intervention.patch.trace,
                        observed_document_ids=retrieval_evidence.observed_ids)
                    row["downstream_metadata"] = {"prediction_kind": "native_retrieval_trace",
                        "protocol": "observed_rank.v1", "ranked_document_ids": list(prediction.scores),
                        "added_model_calls": 0, "added_retrieval_calls": 0}
                elif is_patch:
                    backend = (ObservedLLMBackend(captured, monitor, ledger, run_id,
                                                  base_context.pricing_catalog)
                               if captured is not None else None)
                    if history_hook is not None:
                        from agentinputlab.static_history_configuration import QAHistoryBackend

                        backend = QAHistoryBackend(backend, history_hook, state=state, ledger=ledger)
                    retriever = (ObservedRetriever(case_context.retriever, monitor, ledger, run_id)
                                 if case_context.retriever is not None else None)
                    observed_runtime = StaticRuntimeComponents(backend=backend, retriever=retriever)
                    remaining_state = replace(transformed, budget=_remaining(config.budget, ledger))
                    if is_bfcl:
                        downstream = bfcl_configuration.predict(remaining_state, backend=backend,
                            functions=bfcl_functions, settings=bfcl_settings, seed=config.seed)
                    elif is_longbench:
                        if longbench_settings.get("preserve_original_question"):
                            remaining_state, projection = longbench_configuration.project_native_question(
                                reformer_original if reformer_original is not None else state,
                                remaining_state, retriever=retriever,
                                top_k=longbench_settings.get("rewrite_retrieval_top_k"),
                                prefilled=state if reformer_original is not None else None)
                            remaining_state = replace(remaining_state, budget=_remaining(config.budget, ledger))
                            row["longbench_question_projection"] = projection
                        downstream = longbench_configuration.predict(remaining_state, adapter=adapter,
                            backend=backend, settings=longbench_settings, seed=config.seed)
                    elif patch_retrieval is not None:
                        from agentinputlab.benchmarks.ir_query import (
                            IR_QUERY_ENDPOINT_ID,
                            ContextIRQueryPredictor,
                        )
                        from agentinputlab.patch_retrieval_configuration import generate_query

                        if patch_retrieval.get("protocol") == IR_QUERY_ENDPOINT_ID:
                            downstream = ContextIRQueryPredictor(backend, retriever,
                                max_tokens=patch_retrieval["max_tokens"], top_k=retrieval_options["top_k"],
                                seed=config.seed, allow_json_fence=patch_retrieval["allow_json_fence"]).predict(remaining_state)
                        else:
                            query, reader_metadata = generate_query(remaining_state, backend=backend,
                                settings=patch_retrieval, seed=config.seed)
                            query_state = replace(remaining_state, query=query, instruction="", context=(),
                                                  budget=_remaining(config.budget, ledger))
                            downstream = builder.build(replace(request, runtime=observed_runtime)).predict(query_state)
                            downstream = replace(downstream, metadata={**dict(downstream.metadata),
                                                                       "patch_query_reader": reader_metadata})
                    else:
                        downstream = builder.build(replace(request, runtime=observed_runtime)).predict(remaining_state)
                    prediction = downstream.prediction
                    row["downstream_metadata"] = to_jsonable(downstream.metadata)
                elif archived_retrieval is not None:
                    from agentinputlab.archived_retrieval_configuration import (
                        prediction as archived_prediction,
                    )

                    prediction, metadata = archived_prediction(archived_retrieval, state=state,
                        method=method, adapter=adapter, answer=intervention.result.final_answer)
                    row["downstream_metadata"] = to_jsonable(metadata)
                elif composite_retrieval is not None:
                    from agentinputlab.benchmarks.ir_query import parse_ir_query

                    # Decode the architecture's own answer. Never add a reader
                    # call or silently replace it with the original query.
                    allow_fence = composite_retrieval.get("allow_json_fence", False)
                    query = parse_ir_query(intervention.result.final_answer, allow_json_fence=allow_fence)
                    observed_retriever = ObservedRetriever(case_context.retriever, monitor, ledger, run_id)
                    query_state = replace(state, query=query, instruction="", context=(),
                                          budget=_remaining(config.budget, ledger))
                    observed_runtime = StaticRuntimeComponents(backend=None, retriever=observed_retriever)
                    downstream = builder.build(replace(request, runtime=observed_runtime)).predict(query_state)
                    prediction = downstream.prediction
                    row["downstream_metadata"] = {
                        **to_jsonable(downstream.metadata), "composite_answer_no_extra_call": True,
                        "decoder_policy": "single_json_fence.v1" if allow_fence else "strict_json_query.v1",
                    }
                elif is_longbench:
                    prediction, metadata = longbench_configuration.composite_prediction(
                        intervention.result.final_answer, state=state, benchmark_id=benchmark_id,
                        settings=longbench_settings)
                    row["downstream_metadata"] = to_jsonable(metadata)
                else:
                    # No second agent call: score the architecture's own final answer.
                    from agentinputlab.experiments.production_builders import (
                        _parse_predictor_parameters,
                    )

                    qa = _parse_predictor_parameters(parameters)["qa"]
                    prediction, audit = _parse_visible_qa_prediction(
                        intervention.result.final_answer, benchmark_id=benchmark_id, state=state,
                        output_format=qa["output_format"], identifier_policy=qa["identifier_policy"],
                        allow_json_fence=qa.get("allow_json_fence", False),
                        preserve_empty_answer=qrag_qa_preserve_empty_answer)
                    row["downstream_metadata"] = {"composite_answer_no_extra_call": True,
                                                   "identifier_audit": to_jsonable(audit)}
                    if qrag_qa_preserve_empty_answer:
                        row["downstream_metadata"]["empty_answer_policy"] = "qrag_instruction_native_string.v1"
                ledger.require_within_budget()
                stage = "score"
                scored = build_benchmark_example_record(adapter, example, state_hash=state.state_hash,
                                                         prediction=prediction)
                row.update(status="scored", scored=True, evaluation=to_jsonable(scored))
        except _AwaitingClarification as pending:
            row.update(status="awaiting_clarification", scored=False, clarification={
                "questions": pending.questions,
                "policy": "static_case_requires_external_answer.v1",
                "downstream_executed": False,
                "automatic_resume": False,
            })
        except Exception as exc:
            # Never retry. Monitor receipts retain known usage even on parse/provider failure.
            row.update(status="failed", scored=False,
                       failure={"type": type(exc).__name__, "stage": stage,
                                "message": "Execution failed; inspect redacted monitor evidence. No automatic retry."})
        row["usage"] = to_jsonable(ledger.usage)
        if retrieval_evidence is not None:
            row["observed_retrieval_ids"] = to_jsonable(retrieval_evidence.calls)
        if history_hook is not None:
            row["live_history_records"] = to_jsonable(history_hook.records)
            row["history_compression_effect_claimed"] = False
        row["model_responses"] = captured.calls if captured is not None else []
        row["monitor"] = to_jsonable(monitor.list_snapshots())
        records.append(row)
        if on_record is not None:
            on_record(to_jsonable(row))
    scored_rows = [r for r in records if r["scored"]]
    keys = set.intersection(*(set(r["evaluation"]["metrics"]) for r in scored_rows)) if scored_rows else set()
    return {"benchmark_id": benchmark_id, "split": split, "release_id": config.release_id, **sample_metadata,
            "selected_example_ids": list(example_ids),
            "seed": config.seed, "budget_per_case": to_jsonable(config.budget),
            "selected_count": len(records), "scored_count": len(scored_rows),
            "failed_count": sum(r["status"] == "failed" for r in records),
            "awaiting_clarification_count": sum(r["status"] == "awaiting_clarification" for r in records),
            "automatic_retry": False,
            "metrics_over_scored_cases": {key: sum(r["evaluation"]["metrics"][key]
                                                    for r in scored_rows) / len(scored_rows) for key in sorted(keys)},
            "failure_score_policy": "no fabricated metric; failures retained in selected_count and records",
            "records": records}
