"""Deterministic provider used only for transport and receipt tests."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..protocol import (
    ProviderArmResult,
    content_hash,
    materialized_state,
)


class FixtureProvider:
    benchmark_id = "fixture.common_support"
    source_revision = "0" * 40
    scorer_id = "fixture.official_scorer"
    scorer_revision = "b" * 64

    def doctor(self, config: Mapping[str, Any]) -> tuple[str, ...]:
        return (
            ()
            if config.get("fixture") is True
            and config.get("locked_dependency_hashes") == {"official_runtime": "1" * 64}
            else ("fixture and locked dependencies are required",)
        )

    def capture(
        self,
        config: Mapping[str, Any],
        *,
        output_root: Path,
    ) -> tuple[Mapping[str, Any], ...]:
        del output_root
        if self.doctor(config):
            return ()
        return (
            {
                "checkpoint_id": "fixture-1",
                "group_key": "fixture-group-1",
                "visible_state": {
                    "query": "fixture query",
                    "task_id": "fixture-1",
                    "instruction": "fixture instruction",
                    "context": [],
                    "messages": [],
                    "tool_names": [],
                    "metadata": {},
                    "budget": {
                        "max_model_calls": 4,
                        "max_retrieval_calls": 4,
                        "max_tool_calls": 4,
                        "max_output_tokens": 128,
                        "max_context_tokens": None,
                        "max_cost_usd": None,
                    },
                },
                "eligible_families": [
                    "reformulate",
                    "decompose",
                    "acquire",
                    "clarify",
                    "curate",
                ],
                "protected": {
                    "fixture": True,
                    "locked_dependency_hashes": {
                        "official_runtime": "1" * 64,
                    },
                },
            },
        )

    def run_arm(
        self,
        config: Mapping[str, Any],
        *,
        replay_record: Mapping[str, Any],
        arm_payload: Mapping[str, Any],
        seed: int,
        role: str,
        work_dir: Path,
    ) -> ProviderArmResult:
        del config, work_dir
        state = materialized_state(arm_payload)
        score = float(len(str(state["query"])))
        trace = {
            "provider": "fixture",
            "checkpoint_id": replay_record["checkpoint_id"],
            "role": role,
            "query_hash": content_hash(state["query"]),
            "seed": seed,
        }
        return ProviderArmResult(
            metrics={"official_success": score},
            trace=trace,
            usage={"tool_calls": 1},
            runtime_instance_id=content_hash(
                {
                    "provider": "fixture",
                    "checkpoint_id": replay_record["checkpoint_id"],
                    "role": role,
                    "seed": seed,
                }
            ),
        )


def factory() -> FixtureProvider:
    return FixtureProvider()
