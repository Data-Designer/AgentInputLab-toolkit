"""Fixed fresh QA online streams; explicit feedback, never offline/paper evidence."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from run_clariq_acceptance import verify_services
from run_static_acceptance import QA_CELLS, digest, save

from agentinputlab import ExecutionBudget
from agentinputlab.api import create_intervention
from agentinputlab.benchmarks import load_production_benchmark_adapter
from agentinputlab.benchmarks.qa import QAPrediction

SOURCE_INDICES = [216, 217]


def stream_example(example):
    state = example.visible_state(budget=ExecutionBudget(max_model_calls=8, max_output_tokens=4096))
    # Explicit online feedback is never inserted into generator-visible input.
    target = example.evaluator_gold["reference_groups"][0]["answers"][0]
    return dict(
        example_id=example.example_id,
        input=dict(
            question=state.query,
            context="\n\n".join(f"[{b.block_id}] {b.text}" for b in state.context),
        ),
        expected_output=target,
        metadata=dict(
            visible_state_hash=state.state_hash,
            feedback_kind="source_first_reference_answer",
            online_feedback_not_independent_heldout=True,
        ),
    )


def run(config, output):
    assert len(config["models"]) == 1 and not output.exists()
    model = config["models"][0]
    binding = verify_services([model])[model["id"]]
    fp = dict(model_id=model["id"], revision=digest(binding["model_json_hashes"]), server="vllm")
    roles = {r: dict(fp) for r in ("target", "reflector", "curator")}
    startup = dict(
        api_provider="openai",
        max_tokens_per_call=512,
        json_mode=True,
        base_url=model["base_url"],
        api_key_env="OPENAI_API_KEY",
        extra_body=model["extra_body"],
        roles={
            r: dict(
                model=model["id"],
                fingerprint=roles[r],
                input_cost_per_million=0.0,
                output_cost_per_million=0.0,
            )
            for r in roles
        },
    )
    output.mkdir()
    (output / "terminals").mkdir()
    (output / "claims").mkdir()
    adapters = {
        cell: load_production_benchmark_adapter(cell, config["prepared_root"])
        for cell, _ in QA_CELLS
    }
    selected = {
        cell: sorted(adapters[cell].examples(split), key=lambda x: x.example_id)[216:218]
        for cell, split in QA_CELLS
    }
    assert all(len(v) == 2 for v in selected.values())
    streams = {cell: [stream_example(e) for e in examples] for cell, examples in selected.items()}
    save(
        output / "manifest.json",
        dict(
            release="recipe.ace.online_transductive",
            model=model,
            model_binding=binding,
            source_indices=SOURCE_INDICES,
            streams=streams,
            seed=17,
            expected_positions=8,
            stream_count=4,
            max_physical_provider_requests=64,
            per_case_output_budget=4096,
            per_call_ceiling=512,
            per_case_call_budget=8,
            initial_and_window_and_feedback_generation_in_same_budget=True,
            feedback_explicitly_allowed=True,
            not_offline_or_independent_heldout=True,
            dataset_files={cell: dict(a.prepared.file_sha256) for cell, a in adapters.items()},
            paper_results_changed=False,
            automatic_retry=False,
        ),
    )
    totals = Counter()
    for cell, examples in selected.items():
        key = digest(
            dict(
                model=model["id"],
                benchmark=cell,
                indices=SOURCE_INDICES,
                release="recipe.ace.online_transductive",
            )
        )
        save(
            output / "claims" / (key + ".json"),
            dict(
                benchmark=cell,
                source_case_ids=[e.example_id for e in examples],
                stream=streams[cell],
                model=model,
            ),
        )
        recipe = create_intervention(
            "recipe.ace.online_transductive", controls=dict(seed=17, max_rounds=1, window_size=1)
        )
        result = recipe.run_stream(
            streams[cell],
            checkout=config["ace_checkout"],
            llm_config=startup,
            output_root=output / ("stream-" + key),
            allow_test_feedback=True,
            source_and_asset_acknowledged=True,
        )
        outcomes = []
        for example, primary in zip(examples, result.terminal["primary_outcomes"], strict=True):
            valid = primary["status"] == "native_prediction_returned"
            answer = primary["native_prediction"]["final_answer"] if valid else ""
            metrics = dict(adapters[cell].score(example, QAPrediction(answer=answer)))
            row = dict(
                benchmark=cell,
                case_id=example.example_id,
                status=primary["status"],
                native_primary=primary,
                metrics=metrics,
                score_kind="native_QA_primary_preupdate"
                if valid
                else "statistical_missing_prediction_preserving_raw_failure",
                artifact_id=result.artifact_id,
                per_case_usage=result.terminal["per_case_usage"].get(example.example_id, {}),
                feedback_transductive_not_heldout=True,
                automatic_retry=False,
            )
            save(
                output / "terminals" / (digest(dict(cell=cell, case=example.example_id)) + ".json"),
                row,
            )
            outcomes.append(row)
            totals[primary["status"]] += 1
        save(
            output / ("stream-" + key) / "independent-native-QA.json",
            dict(
                benchmark=cell,
                outcomes=outcomes,
                fixed_failure_denominator=2,
                no_model_calls_in_scoring=True,
            ),
        )
        if result.terminal["status"] == "execution_failed_before_model":
            raise RuntimeError(
                "zero-model-call prerequisite failure; inspect before any subsequent stream"
            )
    rows = [json.loads(p.read_text()) for p in (output / "terminals").glob("*.json")]
    assert len(rows) == 8
    save(
        output / "complete.json",
        dict(
            expected_positions=8,
            terminal_positions=8,
            primary_statuses=dict(totals),
            physical_provider_requests=sum(r["per_case_usage"].get("model_calls", 0) for r in rows),
            physical_provider_input_tokens=sum(
                r["per_case_usage"].get("input_tokens", 0) for r in rows
            ),
            physical_provider_output_tokens=sum(
                r["per_case_usage"].get("output_tokens", 0) for r in rows
            ),
            initial_window_adaptation_costs_included=True,
            feedback_transductive_not_heldout=True,
            automatic_retry=False,
            publication_ready=False,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("operator must supply explicit local deployment key environment")
    run(json.loads(args.config.read_text()), args.output)
