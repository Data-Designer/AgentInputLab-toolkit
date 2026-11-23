"""Built-in SkillOpt environment binding; fixtures are not model evidence."""

import hashlib
import json

import pytest

from agentinputlab.contracts import ValidationError
from agentinputlab.fit_configuration import FitConfig
from tests.unit.test_fit_configuration import skill
from tests.unit.test_retrieval_fit_configuration import inputs


def configured(tmp_path, monkeypatch):
    _, feedback, adapter = inputs(tmp_path, monkeypatch)
    value = skill(tmp_path)
    identity = dict(model_id="m", revision="r", server="vllm")
    value["model_fingerprints"]["target"] = identity
    value["environment"] = dict(
        kind="retrieval",
        source_input=feedback["source_input"],
        backend=dict(
            model_id="m",
            base_url="http://localhost:9/v1",
            api_key_env="TEST_SKILLOPT_KEY",
            deployment_identity=dict(identity),
        ),
        per_call_output_tokens=512,
        seed=17,
        stop_policy="provider",
        partition_label="declared_compatibility_not_quality",
    )
    return value, adapter


def test_skillopt_builtin_environment_needs_no_caller_code(tmp_path, monkeypatch):
    from agentinputlab.recipes.skillopt import SkillOptOfflineRecipe
    from tests.unit.test_skillopt_common_support import _artifact

    value, _ = configured(tmp_path, monkeypatch)
    monkeypatch.setenv("TEST_SKILLOPT_KEY", "local-only-not-in-config")
    seen = {}

    def fit(self, **kwargs):
        seen["binding"] = self.environment_binding
        seen["environment"] = kwargs["environment"]
        return _artifact()

    monkeypatch.setattr(SkillOptOfflineRecipe, "fit", fit)
    result = FitConfig.from_dict(value).execute(tmp_path / "fit")
    assert result["status"] == "fit_completed"
    binding = seen["binding"]
    assert binding.factory_path.name == "skillopt_builtin_environment.py"
    assert (
        binding.verify()["configuration_sha256"]
        == hashlib.sha256(binding.configuration_path.read_bytes()).hexdigest()
    )
    assert json.loads(binding.configuration_path.read_text()) == value["environment"]
    assert "local-only-not-in-config" not in binding.configuration_path.read_text()


def test_builtin_retrieval_uses_original_adapter_and_groups(tmp_path, monkeypatch):
    import agentinputlab.recipes.skillopt_retrieval as native
    from agentinputlab.recipes.skillopt_builtin_environment import create

    value, adapter = configured(tmp_path, monkeypatch)
    seen = {}

    def build(**kwargs):
        seen.update(kwargs)
        return "original-adapter"

    monkeypatch.setattr(native, "create_skillopt_retrieval_adapter", build)
    assert (
        create(cfg={"minibatch_size": 1}, configuration=value["environment"]) == "original-adapter"
    )
    assert seen["adapter"] is adapter
    assert [e.example_id for e in seen["train"]] == ["train-id"]
    assert [e.example_id for e in seen["validation"]] == ["validation-id"]
    assert seen["seed"] == 17 and seen["per_call_output_tokens"] == 512
    assert seen["cfg"] == {"minibatch_size": 1}
    assert seen["target_backend"].model_id == "m"


def test_builtin_source_relative_path(tmp_path, monkeypatch):
    value, _ = configured(tmp_path, monkeypatch)
    value["environment"]["source_input"]["path"] = "source.json"
    path = tmp_path / "fit.json"
    path.write_text(json.dumps(value))
    config = FitConfig.from_file(path)
    assert config.value["environment"]["source_input"]["path"] == str(tmp_path / "source.json")


@pytest.mark.parametrize(
    "change", ["seed", "model", "tokens", "kind", "code", "secret", "stop", "key"]
)
def test_builtin_rejects_ambiguous_or_invalid_environment(tmp_path, monkeypatch, change):
    value, _ = configured(tmp_path, monkeypatch)
    env = value["environment"]
    if change == "seed":
        env["seed"] = 18
    elif change == "model":
        env["backend"]["deployment_identity"]["revision"] = "other"
    elif change == "tokens":
        env["per_call_output_tokens"] = 0
    elif change == "kind":
        env["kind"] = "tooltalk"
    elif change == "code":
        value["environment_binding"] = {}
    elif change == "secret":
        env["backend"]["api_key"] = "plaintext"
    elif change == "stop":
        env["stop_policy"] = "unknown"
    elif change == "key":
        value["environment_names"] = []
    with pytest.raises(ValidationError):
        FitConfig.from_dict(value)


def test_builtin_source_checksum_before_any_fit(tmp_path, monkeypatch):
    value, _ = configured(tmp_path, monkeypatch)
    value["environment"]["source_input"]["sha256"] = "0" * 64
    monkeypatch.setenv("TEST_SKILLOPT_KEY", "local-only")
    with pytest.raises(Exception, match="checksum"):
        FitConfig.from_dict(value).execute(tmp_path / "fit")
    assert not (tmp_path / "fit").exists()
