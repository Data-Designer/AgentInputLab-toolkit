"""Offline contract tests; native-source/model acceptance is a separate gate."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, replace
from unittest.mock import patch

import pytest

from agentinputlab.contracts import ValidationError
from agentinputlab.recipes.reformer import (
    ReFormeRInductionControls,
    ReFormeRPatternArtifact,
    ReFormeRPatternInduction,
    ReFormeRQueryPair,
    validate_induction_splits,
)
from agentinputlab.registry import AlgorithmRegistry
from workers.reformer.driver import (
    BackendConfig,
    CompatibilityPolicy,
    CompletionResult,
    DriverError,
)
from workers.reformer.induction import _native_induction

PAIR = ReFormeRQueryPair("train-1", "Mercury year", "length of a year on planet Mercury")
HELD = ReFormeRQueryPair("test-1", "Dune author", "who wrote Dune", split="test")
PATTERN = {
    "pattern_name": "Clarify",
    "description": "Resolve ambiguity",
    "transformation_rule": "State the intended entity",
    "examples": [],
}
CONFIG = BackendConfig.from_json(
    json.dumps(
        {
            "base_url": "http://127.0.0.1:8787/v1",
            "api_key_env": "TEST_KEY",
            "model": "fixture",
            "fingerprint": {"model": "fixture"},
            "extra_body": {},
            "timeout_seconds": 30,
            "max_input_tokens_per_call": 100,
            "max_output_tokens": 512,
            "temperature": 0,
            "input_cost_per_million": 0,
            "output_cost_per_million": 0,
        }
    )
)


@dataclass
class NativePairFixture:
    original_query: str
    reformulated_query: str
    query_id: str


@dataclass
class PatternFixture:
    pattern_name: str
    description: str
    transformation_rule: str
    examples: list


class ExtractorFixture:
    def extract_patterns_from_batch(self, pairs, number):
        del number
        messages = [
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "pairs": [vars(pair) for pair in pairs],
                        "previous": [vars(pattern) for pattern in self.consolidated_patterns],
                    }
                ),
            }
        ]
        text = self.call_llm(messages)["choices"][0]["message"]["content"]
        try:
            return [PatternFixture(**item) for item in json.loads(text)]
        except (ValueError, TypeError):
            return []


class Backend:
    def __init__(self, outputs=None):
        self.outputs = outputs if outputs is not None else [json.dumps([PATTERN])]
        self.calls = []

    def fingerprint(self):
        return CONFIG.fingerprint

    def complete(self, **kwargs):
        index = len(self.calls)
        self.calls.append(kwargs)
        result = self.outputs[index]
        if isinstance(result, Exception):
            raise result
        return CompletionResult(result, 50, 25, "stop")


@pytest.fixture
def fit_args(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    policy = CompatibilityPolicy(
        commit="72e52450a922bc0051b5b39c3a6307186a555137",
        license_status="NOASSERTION",
        license_file_present=False,
        license_evidence_path="README.md",
        license_evidence_sha256="a" * 64,
        require_clean_checkout=False,
        source_hashes={
            "reformer/patterns/iterative_pattern_extraction.py": "a" * 64,
            "reformer/patterns/query_reformulation_prompts.py": "b" * 64,
            "reformer/prompts.json": "c" * 64,
        },
        profiles={},
    )
    with (
        patch("workers.reformer.induction.verify_checkout", return_value=source),
        patch("workers.reformer.induction._native_induction") as native,
    ):
        native.return_value.__enter__.return_value = (ExtractorFixture, NativePairFixture)
        yield dict(
            checkout=source,
            training_pairs=[PAIR],
            held_out_pairs=[HELD],
            output_root=tmp_path / "fit",
            backend_config=CONFIG,
            backend=Backend(),
            source_license_acknowledged=True,
            policy=policy,
        )


def test_registry_resolves_training_pipeline():
    assert (
        AlgorithmRegistry.default().resolve_entrypoint("training.reformer.pattern_induction")
        is ReFormeRPatternInduction
    )


def test_fit_reload_and_training_only_prompt(fit_args):
    artifact = ReFormeRPatternInduction().fit(**fit_args)
    assert ReFormeRPatternArtifact.load(artifact.root) == artifact
    assert artifact.patterns[0]["pattern_name"] == "Clarify"
    assert "test-1" not in json.dumps(fit_args["backend"].calls)
    assert fit_args["backend"].calls[0]["max_output_tokens"] == 512
    terminal = json.loads((artifact.root / "fit-terminal.json").read_text())
    assert terminal["usage"]["model_calls"] == 1
    assert terminal["held_out_acceptance_complete"] is False
    assert (artifact.root / "batch-0001/request.json").is_file()


def test_iterative_replacement_seed_and_cost(fit_args):
    second = replace(PAIR, query_id="train-2", original_query="Saturn rings")
    fit_args.update(training_pairs=[PAIR, second], backend=Backend([json.dumps([PATTERN])] * 2))
    artifact = ReFormeRPatternInduction(ReFormeRInductionControls(batch_size=1)).fit(**fit_args)
    calls = fit_args["backend"].calls
    assert [call["seed"] for call in calls] == [17, 18]
    assert json.loads(calls[1]["messages"][0]["content"])["previous"][0] == PATTERN
    assert json.loads((artifact.root / "artifact.json").read_text())["usage"]["model_calls"] == 2


@pytest.mark.parametrize(
    "text", ["", "not json", "[]", json.dumps([dict(PATTERN, description="")])]
)
def test_invalid_final_fit_remains_failure_without_artifact(fit_args, text):
    fit_args["backend"] = Backend([text])
    with pytest.raises(DriverError):
        ReFormeRPatternInduction().fit(**fit_args)
    terminal = json.loads((fit_args["output_root"] / "fit-terminal.json").read_text())
    assert terminal["status"] == "failed_fit"
    assert terminal["usage"]["model_calls"] == 1
    assert not (fit_args["output_root"] / "artifact.json").exists()
    with pytest.raises((ValidationError, FileNotFoundError)):
        ReFormeRPatternArtifact.load(fit_args["output_root"])


def test_provider_failure_has_consumed_cost_and_no_retry(fit_args):
    fit_args["backend"] = Backend(
        [DriverError("BACKEND_CALL_FAILED", "unavailable", consumed=True)]
    )
    with pytest.raises(DriverError):
        ReFormeRPatternInduction().fit(**fit_args)
    terminal = json.loads((fit_args["output_root"] / "fit-terminal.json").read_text())
    assert terminal["status"] == "failed_fit"
    assert terminal["provider_usage_unknown"] is True
    assert len(fit_args["backend"].calls) == 1


def test_insufficient_budget_does_not_dispatch(fit_args):
    with pytest.raises(DriverError, match="remaining token budget"):
        ReFormeRPatternInduction(ReFormeRInductionControls(max_tokens=100)).fit(**fit_args)
    assert not fit_args["backend"].calls


def test_existing_root_is_never_resumed_or_overwritten(fit_args):
    ReFormeRPatternInduction().fit(**fit_args)
    with pytest.raises(FileExistsError):
        ReFormeRPatternInduction().fit(**fit_args)
    assert len(fit_args["backend"].calls) == 1


def test_artifact_tampering_rejected(fit_args):
    artifact = ReFormeRPatternInduction().fit(**fit_args)
    artifact.pattern_path.write_text(json.dumps([dict(PATTERN, pattern_name="Changed")]))
    with pytest.raises(ValidationError, match="identity"):
        ReFormeRPatternArtifact.load(artifact.root)


def test_source_acknowledgement_is_explicit(fit_args):
    fit_args["source_license_acknowledged"] = False
    with pytest.raises(ValidationError):
        ReFormeRPatternInduction().fit(**fit_args)
    assert not fit_args["output_root"].exists()


@pytest.mark.parametrize(
    "held",
    [
        replace(HELD, query_id=PAIR.query_id),
        replace(HELD, original_query="  MERCURY   YEAR "),
        replace(HELD, split="train"),
    ],
)
def test_split_leakage_rejected(held):
    with pytest.raises(ValidationError):
        validate_induction_splits([PAIR], [held])


def test_missing_held_out_declaration_rejected():
    with pytest.raises(ValidationError):
        validate_induction_splits([PAIR], [])


@pytest.mark.parametrize("split", ["validation", "test"])
def test_held_out_queries_need_no_unused_target_rewrites(fit_args, split):
    held = ReFormeRQueryPair(HELD.query_id, HELD.original_query, split=split)
    assert held.reformulated_query is None
    fit_args["held_out_pairs"] = [held]
    artifact = ReFormeRPatternInduction().fit(**fit_args)
    assert ReFormeRPatternArtifact.load(artifact.root) == artifact
    assert "test-1" not in json.dumps(fit_args["backend"].calls)
    audit = json.loads((artifact.root / "artifact.json").read_text())["split_audit"]
    assert audit["held_out_ids"] == [HELD.query_id]
    assert audit["held_out_used_in_model_prompts"] is False


def test_optional_held_out_target_does_not_change_training_calls(fit_args):
    first_backend = fit_args["backend"]
    first = ReFormeRPatternInduction().fit(**fit_args)
    second_backend = Backend()
    fit_args.update(
        output_root=fit_args["output_root"].with_name("fit-without-held-out-label"),
        held_out_pairs=[replace(HELD, reformulated_query=None)],
        backend=second_backend,
    )
    second = ReFormeRPatternInduction().fit(**fit_args)
    assert second_backend.calls == first_backend.calls
    assert second.pattern_sha256 == first.pattern_sha256


@pytest.mark.parametrize("value", [None, "", "  ", 12, "bad\x00target"])
def test_training_rewrite_stays_required(value):
    with pytest.raises(ValidationError):
        ReFormeRQueryPair("train", "original", value, split="train")


def test_held_out_without_target_still_checks_split_overlap():
    held = ReFormeRQueryPair("held", "  MERCURY   YEAR ", split="test")
    with pytest.raises(ValidationError, match="overlap"):
        validate_induction_splits([PAIR], [held])


@pytest.mark.parametrize(
    "settings",
    [{"seed": True}, {"batch_size": 0}, {"max_cost_usd": float("nan")}, {"max_tokens": 1.5}],
)
def test_invalid_controls_rejected(settings):
    with pytest.raises(ValidationError):
        ReFormeRInductionControls(**settings)


def test_import_bindings_restored_on_native_import_failure(tmp_path):
    old = sys.modules.get("ollama")
    with pytest.raises((OSError, ImportError)), _native_induction(tmp_path):
        raise AssertionError("must not reach")
    assert sys.modules.get("ollama") is old
