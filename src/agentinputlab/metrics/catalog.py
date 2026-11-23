"""Machine-readable ownership and readiness of benchmark metric names."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    name: str
    owner: str
    status: str
    implementation: str
    reference: str
    note: str = ""


_IR_MEASURES_REFERENCE = "MacAvaney et al., ECIR 2022; https://ir-measur.es/"
_ROUGE_REFERENCE = "Google Research rouge-score 0.1.2"
_QA_REFERENCE = "SQuAD/Hotpot-style normalized English token overlap"

METRIC_CATALOG: tuple[MetricDefinition, ...] = (
    MetricDefinition("ndcg", "third_party", "implemented", "ir_measures", _IR_MEASURES_REFERENCE),
    MetricDefinition(
        "ndcg_at_10", "third_party", "implemented", "ir_measures", _IR_MEASURES_REFERENCE
    ),
    MetricDefinition("map", "third_party", "implemented", "ir_measures", _IR_MEASURES_REFERENCE),
    MetricDefinition(
        "recall", "benchmark_adapter", "implemented",
        "ir_measures for retrieval; ToolTalk official conversation evaluator for ToolTalk",
        _IR_MEASURES_REFERENCE + "; ToolTalk pinned ToolExecutor",
        "The benchmark/scorer identity disambiguates document recall from tool-call recall.",
    ),
    *(
        MetricDefinition(
            name, "benchmark_adapter", "external_required",
            "workers.official_arenas.providers.tooltalk:ToolTalkProvider",
            "ToolTalk e05f4ce6132c80ed33392b81535b077d56ab28fd ToolExecutor.evaluate_predictions",
            "Requires the pinned official runtime and scorer; registration is not an acceptance result.",
        )
        for name in ("official_success", "precision", "action_precision", "bad_action_rate", "soft_success")
    ),
    MetricDefinition(
        "adapter_hard_stops", "benchmark_adapter", "implemented",
        "workers.official_arenas.providers.tooltalk:_Predictor",
        "ToolTalk adapter generation controls",
        "Diagnostic stop count, distinct from native official_success; not a task-success metric.",
    ),
    MetricDefinition("rouge", "third_party", "implemented", "rouge_score", _ROUGE_REFERENCE),
    MetricDefinition(
        "answer_f1",
        "first_party",
        "implemented",
        "english_qa_f1 + best_reference_score",
        _QA_REFERENCE,
        "A benchmark adapter must still freeze reference selection and language-specific rules.",
    ),
    MetricDefinition(
        "support_f1",
        "first_party",
        "implemented",
        "set_f1",
        "Exact set precision/recall/F1",
        "The adapter defines the canonical support identifier.",
    ),
    MetricDefinition(
        "evidence_f1",
        "first_party",
        "implemented",
        "set_f1",
        "Exact set precision/recall/F1",
        "The adapter defines the canonical evidence identifier.",
    ),
    MetricDefinition(
        "joint_f1",
        "benchmark_adapter",
        "implemented",
        "dataset-specific answer/support/evidence composition",
        "HotpotQA and 2WikiMultiHopQA official evaluators",
        "The adapter profile fixes categorical-answer, alias, and joint precision/recall rules.",
    ),
    MetricDefinition(
        "exact_match",
        "benchmark_adapter",
        "implemented",
        "english_qa_exact_match for QA; raw first-code-line equality for RepoBench diagnostic",
        _QA_REFERENCE + "; AgentInputLab additional code diagnostic",
        "The benchmark identity fixes normalization; RepoBench code_metric remains its official primary metric.",
    ),
    MetricDefinition(
        "passage_id_accuracy",
        "benchmark_adapter",
        "adapter_required",
        "dataset-specific exact ID parser",
        "LongBench official evaluator",
    ),
    MetricDefinition(
        "clarification_utility",
        "benchmark_adapter",
        "adapter_required",
        "ClariQ official evaluator",
        "ClariQ task scorer",
    ),
    MetricDefinition(
        "retrieval_utility",
        "benchmark_adapter",
        "adapter_required",
        "paired frozen-retriever delta",
        "AgentInputLab estimand",
    ),
    MetricDefinition(
        "intent_preservation",
        "benchmark_adapter",
        "adapter_required",
        "contract/annotation evaluator",
        "AgentInputLab estimand",
        "Token recall is a diagnostic proxy, not the confirmatory intent metric.",
    ),
    MetricDefinition(
        "constraint_preservation",
        "benchmark_adapter",
        "adapter_required",
        "task constraint evaluator",
        "AgentInputLab estimand",
    ),
    MetricDefinition(
        "query_fidelity",
        "benchmark_adapter",
        "adapter_required",
        "task query evaluator",
        "AgentInputLab estimand",
    ),
    MetricDefinition(
        "code_metric",
        "benchmark_adapter",
        "adapter_required",
        "LongBench official code scorer",
        "LongBench official evaluator",
    ),
    MetricDefinition(
        "ast_accuracy",
        "external_bridge",
        "caller_asserted_external_bridge",
        "PreparedBFCLV3SimpleAdapter + BFCLV3ScorerBridge",
        "BFCL v1.3 ast_file_runner and ast_checker at ea13468e4423454d0c213704fb87cf7cb3990433",
        "One case-level validity bit from the revision-locked official Python AST path. "
        "The current bridge validates identity and type boundaries but does not execute "
        "or attest the official BFCL runtime, and it is not an aggregate BFCL v3 score.",
    ),
    MetricDefinition(
        "official_reward",
        "external_bridge",
        "caller_asserted_external_bridge",
        "CallerAssertedTau3EvaluationBridge + Tau3EvaluationResult",
        "tau-bench evaluator",
        "The current adapter validates a typed result but does not verify official runtime "
        "capture, continuation, evaluator provenance, or result parity.",
    ),
    MetricDefinition(
        "task_success",
        "external_bridge",
        "caller_asserted_external_bridge",
        "AppWorldULEvaluationResult",
        "AppWorld per-task TestTracker success",
        "The adapter returns the official per-task success bit on a [0, 1] scale. "
        "Task goal completion (TGC) and scenario goal completion (SGC) are computed "
        "later by the official complete-group evaluator, never fabricated per checkpoint.",
    ),
    MetricDefinition(
        "task_completion_pass",
        "external_bridge",
        "caller_asserted_external_bridge",
        "StateBenchEvaluationResult",
        "STATE-Bench official trajectory scorer",
        "One-run completion bit; official pass^5 requires five complete comparable runs.",
    ),
    MetricDefinition(
        "state_requirements_met",
        "external_bridge",
        "caller_asserted_external_bridge",
        "StateBenchEvaluationResult",
        "STATE-Bench deterministic final-state assertions",
    ),
    MetricDefinition(
        "task_requirements_met",
        "external_bridge",
        "caller_asserted_external_bridge",
        "StateBenchEvaluationResult",
        "STATE-Bench task-requirement judge",
    ),
    MetricDefinition(
        "ux_score",
        "external_bridge",
        "caller_asserted_external_bridge",
        "StateBenchEvaluationResult",
        "STATE-Bench UX judge and resource adjustment",
        "Single-trajectory score in [1, 5].",
    ),
    MetricDefinition(
        "cost_usd",
        "external_bridge",
        "caller_asserted_external_bridge",
        "StateBenchEvaluationResult or ALEEvaluationResult",
        "STATE-Bench/ALE agent-reported provider cost",
        "The bridge must preserve the official cost basis; AgentInputLab does not infer "
        "provider pricing from token counts.",
    ),
    MetricDefinition(
        "task_score",
        "external_bridge",
        "caller_asserted_external_bridge",
        "ALEEvaluationResult",
        "Agents' Last Exam official hidden-reference grader",
        "A task-level score in [0, 1] supplied only after the official ALE run stages "
        "hidden references and invokes its grader.",
    ),
    MetricDefinition(
        "wall_seconds",
        "external_bridge",
        "caller_asserted_external_bridge",
        "ALEEvaluationResult",
        "Agents' Last Exam official-run telemetry",
        "Non-negative elapsed wall time supplied by the operator-owned ALE bridge.",
    ),
    MetricDefinition(
        "turns",
        "external_bridge",
        "caller_asserted_external_bridge",
        "StateBenchEvaluationResult",
        "STATE-Bench deterministic efficiency telemetry",
    ),
    MetricDefinition(
        "tool_calls",
        "external_bridge",
        "caller_asserted_external_bridge",
        "StateBenchEvaluationResult",
        "STATE-Bench deterministic efficiency telemetry",
    ),
    MetricDefinition(
        "milestone_similarity",
        "external_bridge",
        "caller_asserted_external_bridge",
        "ToolSandboxEvaluationResult",
        "ToolSandbox official milestone evaluator",
        "Similarity to the scenario milestones before the official minefield rule is applied.",
    ),
    MetricDefinition(
        "minefield_similarity",
        "external_bridge",
        "caller_asserted_external_bridge",
        "ToolSandboxEvaluationResult",
        "ToolSandbox official minefield evaluator",
        "Similarity to forbidden state changes; any non-zero value nullifies official_similarity.",
    ),
    MetricDefinition(
        "official_similarity",
        "external_bridge",
        "caller_asserted_external_bridge",
        "ToolSandboxEvaluationResult",
        "ToolSandbox official milestone/minefield evaluator",
        "Equals milestone_similarity only when minefield_similarity is zero; otherwise zero.",
    ),
    MetricDefinition(
        "turn_count",
        "external_bridge",
        "caller_asserted_external_bridge",
        "ToolSandboxEvaluationResult",
        "ToolSandbox trajectory telemetry",
    ),
    MetricDefinition(
        "harm_rate",
        "first_party_paired_estimator",
        "implemented_composition_utility",
        "summarize_pairs",
        "AgentInputLab paired estimator",
        "Computed outside benchmark adapters from matched baseline/treatment values and an "
        "explicit harm threshold. Confirmatory harm is a study-level event definition, not a "
        "separate single-arm metric.",
    ),
    MetricDefinition(
        "cost",
        "external_bridge_or_runner",
        "composition_required",
        "typed external cost or observed usage/wall-time composition",
        "AgentInputLab run protocol",
        "Tau3, AppWorld-UL, and ToolSandbox accept caller-asserted bridge cost with a "
        "declared cost_basis; the value is not silently inferred from runner telemetry.",
    ),
    MetricDefinition(
        "valid_success",
        "study_protocol",
        "study_composition_required",
        "not computed by the current benchmark adapter",
        "AgentInputLab run protocol",
        "Requires task success plus preregistered contract-preservation and leakage-validity "
        "inputs at the study layer.",
    ),
    MetricDefinition(
        "terminal_success",
        "external_bridge",
        "caller_asserted_external_bridge",
        "ProviderArmResult",
        "tau retail provider terminal accounting",
        "Budget/timeout-aware terminal success supplied by the native retail provider; distinct from official_reward and never inferred from a model response alone.",
    ),
)


def metric_definition(name: str) -> MetricDefinition:
    for definition in METRIC_CATALOG:
        if definition.name == name:
            return definition
    raise KeyError(f"unknown metric: {name}")
