"""Synthetic environment fixtures; no model fitting or native-score evidence."""
from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

from agentinputlab.contracts import ArtifactError, ExternalDependencyError, ValidationError
from agentinputlab.recipes import SkillOptEnvironmentBinding, SkillOptOfflineRecipe
from agentinputlab.recipes.skillopt import load_skillopt_artifact
from agentinputlab.recipes.skillopt_environment import ENVIRONMENT_RECEIPT, main, run_with_environment


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture
def environment(tmp_path, monkeypatch):
    root = tmp_path / "source"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts/train.py").write_text('''
import json,sys
from pathlib import Path
def get_adapter(cfg):
    raise AssertionError("synthetic built-in factory must be replaced explicitly")
def main():
    adapter=get_adapter({"env":"native-ir-fixture","seed":17})
    assert adapter.fingerprint()["native_fixture"]
    Path(sys.argv[sys.argv.index("--out_root")+1],"seen-argv.json").write_text(json.dumps(sys.argv[1:]))
''')
    base = types.ModuleType("skillopt.envs.base")
    base.EnvAdapter = type("EnvAdapter", (), {})
    monkeypatch.setitem(sys.modules, "skillopt.envs.base", base)
    monkeypatch.setattr("agentinputlab.recipes.skillopt.validate_skillopt_checkout", lambda p: root)
    factory = tmp_path / "factory.py"
    factory.write_text('''
from skillopt.envs.base import EnvAdapter
class Fixture(EnvAdapter):
    def fingerprint(self):return {"native_fixture":True,"metric":"original-fixture-metric"}
def create(*,cfg,configuration):
    assert cfg=={"env":"native-ir-fixture","seed":17}
    assert configuration=={"task":"original-fixture-task"}
    return Fixture()
''')
    config = tmp_path / "task.json"
    config.write_text('{"task":"original-fixture-task"}')
    binding = SkillOptEnvironmentBinding(factory, sha(factory), config, sha(config))
    output = tmp_path / "output"
    output.mkdir()
    return root, binding, output


def test_explicit_factory_uses_native_cli_and_preserves_all_native_arguments(environment):
    root, binding, output = environment
    previous_argv, previous_path = list(sys.argv), list(sys.path)
    args = ["--config", "settings.yaml", "--out_root", str(output), "--cfg-options", "train.seed=17"]
    run_with_environment(checkout=root, binding=binding, receipt_path=output / ENVIRONMENT_RECEIPT, native_args=args)
    assert json.loads((output / "seen-argv.json").read_text()) == args
    receipt = json.loads((output / ENVIRONMENT_RECEIPT).read_text())
    assert receipt["binding"] == binding.verify()
    assert receipt["environment_fingerprint"]["metric"] == "original-fixture-metric"
    assert receipt["profile_kind"] == "caller_environment_adapter"
    assert sys.argv == previous_argv and sys.path == previous_path


def test_cli_binding_separates_wrapper_and_unchanged_native_arguments(environment):
    root, binding, output = environment
    assert main(["--checkout", str(root), *binding.command_options(),
                 "--receipt", str(output / ENVIRONMENT_RECEIPT), "--",
                 "--out_root", str(output)]) == 0
    assert json.loads((output / "seen-argv.json").read_text()) == ["--out_root", str(output)]


def test_recipe_default_native_command_remains_unchanged(environment):
    root, binding, output = environment
    legacy = SkillOptOfflineRecipe().build_command(checkout=root, config_path=root / "cfg", output_root=output)
    plugin = SkillOptOfflineRecipe(environment_binding=binding).build_command(
        checkout=root, config_path=root / "cfg", output_root=output)
    assert legacy[1] == str(root / "scripts/train.py")
    assert plugin[1:3] == ("-m", "agentinputlab.recipes.skillopt_environment_cli")
    assert plugin[plugin.index("--") + 1:] == legacy[2:]


@pytest.mark.parametrize("field", ["factory", "configuration"])
def test_changed_pinned_inputs_fail_before_factory_import(environment, field):
    root, binding, output = environment
    path = Path(getattr(binding, field + "_path"))
    path.write_text("changed")
    with pytest.raises((ArtifactError, ExternalDependencyError, ValidationError)):
        run_with_environment(checkout=root, binding=binding, receipt_path=output / ENVIRONMENT_RECEIPT,
                             native_args=["--out_root", str(output)])
    assert not list(output.iterdir())


def test_wrong_adapter_type_fails_without_builtin_or_model_fallback(environment):
    root, binding, output = environment
    p = Path(binding.factory_path)
    p.write_text('def create(**kwargs):return object()')
    binding = SkillOptEnvironmentBinding(p, sha(p), binding.configuration_path, binding.configuration_sha256)
    with pytest.raises(ValidationError, match="native EnvAdapter"):
        run_with_environment(checkout=root, binding=binding, receipt_path=output / ENVIRONMENT_RECEIPT,
                             native_args=["--out_root", str(output)])
    assert not list(output.iterdir())


def test_native_factory_errors_propagate_once(environment):
    root, binding, output = environment
    p = Path(binding.factory_path)
    marker = output / "factory-called"
    p.write_text(f'''from pathlib import Path
def create(**kwargs):
    with Path({str(marker)!r}).open("x") as f:f.write("one call")
    raise RuntimeError("real fixture environment failure")
''')
    binding = SkillOptEnvironmentBinding(p, sha(p), binding.configuration_path, binding.configuration_sha256)
    with pytest.raises(RuntimeError, match="real fixture"):
        run_with_environment(checkout=root, binding=binding, receipt_path=output / ENVIRONMENT_RECEIPT,
                             native_args=["--out_root", str(output)])
    assert marker.read_text() == "one call"
    assert not (output / ENVIRONMENT_RECEIPT).exists()


def test_custom_environment_is_hash_bound_in_fitted_artifact(environment):
    root, binding, output = environment
    run_with_environment(checkout=root, binding=binding, receipt_path=output / ENVIRONMENT_RECEIPT,
                         native_args=["--out_root", str(output)])
    for name, text in (("best_skill.md", "real-fixture-skill"), ("config.json", "{}"),
                       ("history.json", "[]"), ("runtime_state.json", "{}")):
        (output / name).write_text(text)
    kwargs = dict(config_sha256="a" * 64, controls={"seed":17},
                  model_fingerprints={"target":{"id":"fixture"},"optimizer":{"id":"fixture"}},
                  command_sha256="b" * 64)
    plugin = load_skillopt_artifact(output, **kwargs, environment_binding=binding.verify())
    with pytest.raises(ExternalDependencyError, match="explicit environment binding"):
        load_skillopt_artifact(output, **kwargs)
    original = output.with_name("legacy-output")
    original.mkdir()
    for name in ("best_skill.md", "config.json", "history.json", "runtime_state.json"):
        (original / name).write_bytes((output / name).read_bytes())
    legacy = load_skillopt_artifact(original, **kwargs)
    assert plugin.artifact_id != legacy.artifact_id
    assert plugin.source_hashes["operator_environment_factory"] == binding.factory_sha256
    assert plugin.output_hashes[ENVIRONMENT_RECEIPT] == sha(output / ENVIRONMENT_RECEIPT)
    assert ENVIRONMENT_RECEIPT not in legacy.output_hashes
    assert "operator_environment_factory" not in legacy.source_hashes


@pytest.mark.parametrize("field,value", [("factory_sha256","main"),("factory_name","_private"),
                                        ("factory_name","module:create"),("configuration_sha256",None)])
def test_unbound_factory_identity_is_rejected(environment, field, value):
    _, binding, _ = environment
    values = {name:getattr(binding,name) for name in binding.__dataclass_fields__}
    values[field] = value
    with pytest.raises(ValidationError):
        SkillOptEnvironmentBinding(**values)
