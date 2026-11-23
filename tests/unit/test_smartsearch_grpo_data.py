"""Native data/config contracts using explicit offline fixture records."""
import copy
import json
from types import SimpleNamespace

import pytest

from agentinputlab.contracts import ValidationError
from agentinputlab.contracts.serialization import content_hash
from agentinputlab.recipes import (
    SMARTSEARCH_GRPO_ENVIRONMENT,
    SmartSearchGRPOControls,
    SmartSearchGRPODataArtifact,
    SmartSearchGRPOExample,
    SmartSearchTrainingCase,
    prepare_smartsearch_grpo_data,
)


def example(identity="t", question="Fixture training", split="train"):
    return SmartSearchGRPOExample(SmartSearchTrainingCase(identity, question, split), ("fixture gold",))


def prepare(tmp_path, monkeypatch, **kwargs):
    from agentinputlab.recipes import smartsearch_grpo_data as module

    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    monkeypatch.setattr(module, "verify_smartsearch_grpo_source", lambda *_: (source, {"fixture_only": True}))
    values = dict(checkout=source, output_root=tmp_path / "out", train=[example()],
                  validation=[example("d", "Fixture development", "validation")],
                  data_provenance={"fixture_only": True, "model_calls": 0})
    values.update(kwargs)
    return prepare_smartsearch_grpo_data(**values)


def test_native_parquet_fields_disjoint_roles_and_create_only(tmp_path, monkeypatch):
    pq = pytest.importorskip("pyarrow.parquet")
    artifact = prepare(tmp_path, monkeypatch)
    assert artifact.train_rows == artifact.validation_rows == 1
    rows = pq.read_table(artifact.root / "train.parquet").to_pylist()
    assert rows[0]["question"] == "Fixture training" and "fixture gold" not in rows[0]["question"]
    assert rows[0]["reward_model"] == {"style": "rule", "ground_truth": ["fixture gold"]}
    assert rows[0]["extra_info"]["env"] == SMARTSEARCH_GRPO_ENVIRONMENT
    assert json.loads(rows[0]["extra_info"]["func_schemas"])[0]["function"]["name"] == "wikipedia_search"
    metadata = json.loads((artifact.root / "manifest.json").read_text())
    assert metadata["no_native_test_to_dev_downsampling"]
    assert metadata["model_calls"] == metadata["optimizer_steps"] == 0
    assert SmartSearchGRPODataArtifact.load(artifact.root).artifact_id == artifact.artifact_id
    with pytest.raises(ValidationError, match="create-only"):
        prepare(tmp_path, monkeypatch)


@pytest.mark.parametrize("override", [
    {"validation": [example("t", "Distinct development", "validation")]},
    {"validation": [example("d", " fixture TRAINING ", "validation")]},
    {"train": [example("d", "Wrong role", "validation")]},
    {"test": [SmartSearchTrainingCase("test", "Fixture training", "test")]},
    {"train": []}, {"max_rows": 0}, {"max_input_bytes": 1}, {"data_provenance": {}},
])
def test_roles_limits_and_leakage_rejected_before_output(tmp_path, monkeypatch, override):
    with pytest.raises(ValidationError):
        prepare(tmp_path, monkeypatch, **override)
    assert not (tmp_path / "out").exists()


def test_artifact_rejects_changed_parquet_and_false_counts(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    artifact = prepare(tmp_path, monkeypatch)
    manifest = artifact.root / "manifest.json"
    value = json.loads(manifest.read_text())
    value.pop("artifact_id")
    value["train_rows"] = 2
    value["artifact_id"] = content_hash(value)
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValidationError, match="row count"):
        SmartSearchGRPODataArtifact.load(artifact.root)
    path = artifact.root / "train.parquet"
    path.write_bytes(path.read_bytes() + b"fixture change")
    with pytest.raises(ValidationError, match="bytes changed"):
        SmartSearchGRPODataArtifact.load(artifact.root)


def test_native_controls_require_disclosed_variants_and_never_truncate():
    controls = SmartSearchGRPOControls()
    assert controls.planned_steps(512) == 4
    with pytest.raises(ValidationError, match="empty"):
        controls.planned_steps(4)
    with pytest.raises(ValidationError, match="explicit labeled"):
        SmartSearchGRPOControls(train_batch_size=4, ppo_mini_batch_size=2)
    bounded = SmartSearchGRPOControls(train_batch_size=4, ppo_mini_batch_size=2,
                                     gpu_processes=2, bounded_training_variant=True, max_optimizer_steps=2)
    assert bounded.planned_steps(4) == 2
    with pytest.raises(ValidationError, match="actor optimizer plan"):
        bounded.planned_actor_optimizer_steps(4, ppo_epochs=1)
    admitted = SmartSearchGRPOControls(train_batch_size=4, ppo_mini_batch_size=2,
        gpu_processes=2, bounded_training_variant=True, max_optimizer_steps=4)
    assert admitted.planned_actor_optimizer_steps(4, ppo_epochs=1) == 4
    with pytest.raises(ValidationError, match="actor optimizer plan"):
        admitted.planned_actor_optimizer_steps(4, ppo_epochs=2)
    for epochs in (True, 0, 1.5):
        with pytest.raises(ValidationError, match="PPO epochs"):
            admitted.planned_actor_optimizer_steps(4, ppo_epochs=epochs)
    with pytest.raises(ValidationError, match="not truncated"):
        bounded.planned_steps(8)
    for values in ({"rollout_samples": 1}, {"gpu_processes": 5}, {"seed": True},
                   {"gpu_memory_utilization": float("nan")}, {"gpu_processes": 3}):
        with pytest.raises(ValidationError):
            SmartSearchGRPOControls(**values)
    with pytest.raises(ValidationError, match="divisibility"):
        SmartSearchGRPOControls(train_batch_size=6, ppo_mini_batch_size=2,
            rollout_samples=2, gpu_processes=3, bounded_training_variant=True)


def test_fixture_native_config_keeps_key_training_semantics(tmp_path):
    yaml = pytest.importorskip("yaml")
    from agentinputlab.recipes.smartsearch_grpo_data import _native_grpo_config

    original = {"data": {}, "algorithm": {"kl_ctrl": {}}, "reward_model": {},
                "custom_reward_function": {}, "trainer": {},
                "actor_rollout_ref": {"model": {}, "actor": {"optim": {}, "fsdp_config": {}, "ppo_epochs": 1,
                    "checkpoint": {"contents": ["model", "optimizer", "extra"]}},
                    "rollout": {}, "ref": {"fsdp_config": {}}}}
    path = tmp_path / "src/verl/trainer/config/ppo_trainer.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(original))
    before = copy.deepcopy(original)
    config = _native_grpo_config(tmp_path, model=tmp_path / "model", output_root=tmp_path / "run",
        data=SimpleNamespace(root=tmp_path / "data", train_rows=4),
        controls=SmartSearchGRPOControls(train_batch_size=4, ppo_mini_batch_size=2,
                                         gpu_processes=2, bounded_training_variant=True),
        sandbox_url="http://127.0.0.1:12345/fixture-capability", reward_module=tmp_path / "reward.py")
    assert original == before
    actor, rollout = config["actor_rollout_ref"]["actor"], config["actor_rollout_ref"]["rollout"]
    assert actor["optim"]["lr"] == 1e-6 and actor["kl_loss_coef"] == .001
    assert rollout["n"] == 8 and rollout["max_turns"] == 5
    assert config["data"]["max_prompt_length"] == config["data"]["max_response_length"] == 8192
    assert config["trainer"]["total_epochs"] == 2 and config["trainer"]["resume_mode"] == "disable"
    assert actor["checkpoint"]["contents"] == ["model", "optimizer", "extra", "hf_model"]
    assert config["custom_reward_function"]["name"] == "compute_score"
    with pytest.raises(ValidationError, match="loopback"):
        _native_grpo_config(tmp_path, model=tmp_path / "model", output_root=tmp_path / "run",
            data=SimpleNamespace(root=tmp_path / "data", train_rows=4),
            controls=SmartSearchGRPOControls(train_batch_size=4, ppo_mini_batch_size=2,
                                             bounded_training_variant=True),
            sandbox_url="https://example.org/execute", reward_module=tmp_path / "reward.py")
