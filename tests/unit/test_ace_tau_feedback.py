"""Public Tau contract fixtures, not real benchmark or model acceptance."""

import hashlib
import json
from pathlib import Path

import pytest

from agentinputlab.api import _state
from agentinputlab.contracts import ValidationError
from agentinputlab.contracts.serialization import content_hash, to_jsonable
from agentinputlab.recipes import ACETauFeedback, build_ace_tau_feedback_input
from agentinputlab.recipes.ace_native_plan import _save, source_file_pins
from agentinputlab.recipes.ace_tau import SCHEMA, _same_backbone


def pin(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


class FixtureProvider:
    source_revision = "fixture-native-revision"
    scorer_id = "fixture-native-scorer"
    scorer_revision = "fixture-native-scorer-revision"

    def __init__(self, domain):
        self.benchmark_id = "tau3." + domain

    def _validate_config(self, configuration):
        return configuration


def configuration():
    return dict(
        llm_agent="openai/target",
        llm_user="openai/target",
        llm_args_agent=dict(seed=29),
    )


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    import agentinputlab.recipes.ace_native_plan as common
    import agentinputlab.recipes.ace_tau as tau

    monkeypatch.setattr(common, "verify_native_checkout", lambda root, **kw: Path(root))
    monkeypatch.setattr(tau, "verify_native_checkout", lambda root, **kw: Path(root))
    monkeypatch.setattr(tau, "_provider", FixtureProvider)

    def prepare(domain="retail", mode="online", **source_changes):
        root = tmp_path / domain
        root.mkdir()
        source = root / "scorer.py"
        source.write_text("# fixture native identity\n")
        pins = source_file_pins(root, [source])
        schemas = {key: [dict(name="lookup", parameters=dict(type="object"))] for key in ("a", "b")}
        rows = []
        for key, role in (("a", "train"), ("b", "validation")):
            state = _state(
                dict(
                    task_id=key,
                    query="visible user " + key,
                    instruction="Preserve this task policy.",
                    tool_names=["lookup"],
                    context=[dict(block_id="policy", text="visible evidence")],
                    budget=dict(max_tool_calls=2),
                )
            )
            rows.append(
                dict(
                    case_id=key,
                    role=role,
                    state=to_jsonable(state),
                    state_hash=state.state_hash,
                    protected_checkpoint=dict(hidden_gold="DO_NOT_SEND_THIS_TO_MODEL"),
                    native_configuration=configuration(),
                )
            )
        value = dict(
            schema=SCHEMA,
            benchmark="tau3." + domain,
            domain=domain,
            native_source_revision=FixtureProvider.source_revision,
            source_files=pins,
            fitting_rows=rows,
            heldout_ids=["heldout"],
            source_official_split="train" if domain == "retail" else None,
            partition_source="caller-declared-fixture-groups",
            target_model=dict(model_id="target"),
            environment_model=dict(model_id="target"),
            native_function_schemas=schemas,
        )
        value.update(source_changes)
        path = tmp_path / (domain + "-input.json")
        _save(path, value)
        monkeypatch.setattr(tau, "_native_inputs", lambda *args: ({}, {}, schemas, {}, {}, pins))
        return ACETauFeedback(
            checkout=root,
            source_input=pin(path),
            output_root=tmp_path / (domain + "-feedback"),
            mode=mode,
        )

    return prepare


@pytest.mark.parametrize("domain,split", [("retail", "train"), ("banking_knowledge", None)])
def test_frozen_visible_inputs_and_actual_split_provenance(prepared, domain, split):
    p = prepared(domain)
    stream = p.stream_examples()
    p.validate_online_examples([e.as_dict() for e in stream])
    assert len(stream) == 2
    assert all(e.metadata["source_official_split"] == split for e in stream)
    text = json.dumps(stream[0].as_dict()["input"])
    assert "DO_NOT_SEND_THIS_TO_MODEL" not in text
    assert "Preserve this task policy." in text and "visible evidence" in text
    assert "native_function_tools" in text
    assert p.fingerprint()["environment_model"] == p.fingerprint()["target_model"]


def test_invalid_plan_is_not_an_official_zero(prepared):
    p = prepared()
    assert p.score_answer("not JSON", "a") == 0.0
    assert p.events[-1]["native_metrics"] is None
    assert p.events[-1]["native_task_score_NOT_claimed"]
    assert not (p.root / "native-feedback").exists()


def test_valid_native_zero_and_exact_reuse_are_preserved(prepared):
    p = prepared()
    calls = []

    def score(*args):
        calls.append(args)
        return dict(native_metrics=dict(official_reward=0.0))

    p._native_score = score
    output = '{"actions":[],"response":"cannot proceed"}'
    assert p.score_answer(output, "a") == p.score_answer(output, "a") == 0.0
    assert len(calls) == 1 and p.events[-1]["exact_feedback_reuse_NOT_new_model_draw"]
    assert p.events[-1]["prediction_sha256"] == content_hash(output)


def test_native_failure_propagates_without_scoring_or_retry(prepared):
    p = prepared()
    calls = []

    def fail(*args):
        calls.append(args)
        raise RuntimeError("original native tool error")

    p._native_score = fail
    with pytest.raises(RuntimeError, match="original native tool error"):
        p.score_answer('{"actions":[],"response":"answer"}', "a")
    assert len(calls) == 1 and p.events == [] and p.cache == {}
    with pytest.raises(ValidationError, match="outside"):
        p.score_answer("{}", "heldout")


@pytest.mark.parametrize("domain,split", [("retail", None), ("banking_knowledge", "train")])
def test_official_split_is_not_invented_or_discarded(prepared, domain, split):
    with pytest.raises(ValidationError, match="split provenance"):
        prepared(domain, source_official_split=split)


def test_explicit_configured_seed_and_backbone_are_required():
    config = configuration()
    _same_backbone(config, dict(model_id="target"))
    config["llm_args_agent"]["seed"] = 123
    _same_backbone(config, dict(model_id="target"))
    del config["llm_args_agent"]["seed"]
    with pytest.raises(ValidationError, match="integer evaluated-agent seed"):
        _same_backbone(config, dict(model_id="target"))
    config = dict(configuration(), llm_user="openai/other")
    with pytest.raises(ValidationError, match="backbone"):
        _same_backbone(config, dict(model_id="target"))


def test_builder_rejects_overlapping_group_before_native_runtime(tmp_path, monkeypatch):
    import agentinputlab.recipes.ace_tau as tau

    monkeypatch.setattr(tau, "_provider", FixtureProvider)
    monkeypatch.setattr(tau, "verify_native_checkout", lambda root, **kw: Path(root))
    with pytest.raises(ValidationError, match="disjoint fitting"):
        build_ace_tau_feedback_input(
            checkout=tmp_path,
            domain="retail",
            captures=[],
            native_configurations={},
            train_ids=["a"],
            validation_ids=["b"],
            heldout_ids=["a"],
            partition_label="fixture",
            target_model=dict(model_id="target"),
            output_path=tmp_path / "no.json",
        )
    assert not (tmp_path / "no.json").exists()
