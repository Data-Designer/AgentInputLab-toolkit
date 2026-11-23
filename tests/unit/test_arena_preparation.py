"""Preparation contract tests; fixtures do not certify native model execution."""
import json
import sys

import pytest

from agentinputlab import arena_preparation as prep
from agentinputlab.arena_configuration import NativeArenaConfig
from agentinputlab.contracts.errors import ValidationError

BACKEND = {"model_id": "fixture", "base_url": "http://localhost:1/v1",
           "deployment_identity": {"revision": "fixture"}}


@pytest.fixture
def native(tmp_path):
    return NativeArenaConfig("statebench", str(tmp_path / "upstream"),
                             environment_backend=BACKEND)


def test_creation_requires_explicit_network_before_writes(native, tmp_path):
    root = tmp_path / "bundle"
    with pytest.raises(ValidationError, match="allow-network"):
        prep.prepare_arena(native, environment=root, python=sys.executable)
    assert not root.exists()


def test_tooltalk_requires_supported_python_before_install(tmp_path, monkeypatch):
    native = NativeArenaConfig("tooltalk", str(tmp_path / "source"))
    monkeypatch.setattr(prep, "_probe_python", lambda _: [3, 9, 20])
    with pytest.raises(ValidationError, match="3.10.*3.12"):
        prep.prepare_arena(native, environment=tmp_path / "bundle",
                           python=sys.executable, allow_network=True)
    assert not (tmp_path / "bundle").exists()


def test_tooltalk_cache_is_explicit_and_relative_to_config(tmp_path):
    path = tmp_path / "arena.json"
    path.write_text(json.dumps({"arena": "tooltalk", "checkout": "source", "scorer_cache": "cache"}))
    config = NativeArenaConfig.from_file(path)
    assert config.scorer_cache == str(tmp_path / "cache")
    with pytest.raises(ValidationError, match="ToolTalk"):
        NativeArenaConfig("statebench", str(tmp_path), environment_backend=BACKEND, scorer_cache="cache")


def test_tooltalk_preparation_source_is_registered_pin(tmp_path):
    assert prep._source(tmp_path, "tooltalk").commit == "e05f4ce6132c80ed33392b81535b077d56ab28fd"


def test_tooltalk_preparation_uses_cpu_and_pinned_scorer(tmp_path, monkeypatch):
    calls, checks = [], []
    monkeypatch.setattr(prep, "_probe_python", lambda _: [3, 11, 15])
    monkeypatch.setattr(prep.sys, "platform", "linux")
    monkeypatch.setattr(prep, "_run", lambda command, **kw: calls.append((command, kw)) or "")
    monkeypatch.setattr(prep, "_prepare_source", lambda *args, **kw: None)
    monkeypatch.setattr(prep, "_verify_environment", lambda *args, **kw: checks.append(kw) or {})
    native = NativeArenaConfig("tooltalk", str(tmp_path / "source"), scorer_cache=str(tmp_path / "cache"))
    root = tmp_path / "bundle"
    result = prep.prepare_arena(native, environment=root, python=sys.executable, allow_network=True)
    assert result["recipe"] == "tooltalk-isolated.v1"
    assert any("torch==2.11.0+cpu" in command for command, _ in calls)
    assert not any("sent2vec" in command for command, _ in calls)
    scorer_calls = [(command, kw) for command, kw in calls if "scorer-prepare.log" in str(kw.get("log", ""))]
    assert len(scorer_calls) == 1
    assert "12040accade4e8a0f71eabdb258fecc2e7e948be" in str(scorer_calls[0][0])
    assert scorer_calls[0][1]["environment"]["HF_HUB_CACHE"] == native.scorer_cache
    assert NativeArenaConfig.from_file(root / "arena.json").scorer_cache == native.scorer_cache


def test_existing_unowned_directory_preserved(native, tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    sentinel = root / "important"
    sentinel.write_text("keep")
    with pytest.raises(ValidationError, match="incomplete|unrelated"):
        prep.prepare_arena(native, environment=root, python=sys.executable, allow_network=True)
    assert list(root.iterdir()) == [sentinel]


def test_wrong_python_before_writes(native, tmp_path, monkeypatch):
    monkeypatch.setattr(prep, "_probe_python", lambda _: [3, 11, 15])
    with pytest.raises(ValidationError, match="3.12"):
        prep.prepare_arena(native, environment=tmp_path / "bundle",
                           python=sys.executable, allow_network=True)
    assert not (tmp_path / "bundle").exists()


@pytest.fixture
def installed(native, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(prep, "_probe_python", lambda _: [3, 12, 3])
    monkeypatch.setattr(prep, "_implementation_digest", lambda: "fixture-code")
    monkeypatch.setattr(prep, "_prepare_source", lambda source, destination, **kw: destination.mkdir())
    monkeypatch.setattr(prep, "_verify_environment", lambda root, digest, **kw: {
        "python": "3.12.3", "implementation_sha256": digest,
        "dependencies": "native==1\n", "source_commit": "fixture"})
    def run(command, **kwargs):
        calls.append(command)
        return ""
    monkeypatch.setattr(prep, "_run", run)
    root = tmp_path / "bundle"
    result = prep.prepare_arena(native, environment=root, python=sys.executable, allow_network=True)
    return root, result, calls


def test_owned_env_install_preserves_science(native, installed):
    root, result, calls = installed
    assert result["status"] == "installed"
    assert result["model_calls"] == 0
    config = NativeArenaConfig.from_file(root / "arena.json")
    assert config.parameters == native.parameters
    assert config.environment_backend == native.environment_backend
    assert config.max_output_tokens_per_call == native.max_output_tokens_per_call
    assert config.worker_python == str(root / "venv" / "bin" / "python")
    installs = [command for command in calls if "install" in command]
    assert len(installs) == 1
    assert "--python" in installs[0]
    assert str(root / "venv" / "bin" / "python") in installs[0]
    assert "--no-deps" not in installs[0]
    assert "httpx==0.27.2" not in installs[0]
    assert all("--system-site-packages" not in command for command in calls)


def test_completed_reuse_offline_verifies_without_install(native, installed, monkeypatch):
    root, _, _ = installed
    monkeypatch.setattr(prep, "_run", lambda *a, **kw: pytest.fail("must not install again"))
    result = prep.prepare_arena(native, environment=root, python=sys.executable)
    assert result["status"] == "reused"


def test_changed_config_not_reused(native, installed):
    from dataclasses import replace
    root, _, _ = installed
    with pytest.raises(ValidationError, match="different"):
        prep.prepare_arena(replace(native, max_output_tokens_per_call=256),
                           environment=root, python=sys.executable)


def test_changed_environment_not_reused(native, installed, monkeypatch):
    root, _, _ = installed
    monkeypatch.setattr(prep, "_verify_environment", lambda *a, **kw: {"dependencies": "changed"})
    with pytest.raises(ValidationError, match="changed"):
        prep.prepare_arena(native, environment=root, python=sys.executable)


def test_failure_retains_receipt_without_ready_config(native, tmp_path, monkeypatch):
    monkeypatch.setattr(prep, "_probe_python", lambda _: [3, 12, 3])
    def fail(*args, **kwargs):
        raise ValidationError("fixture failure")
    monkeypatch.setattr(prep, "_prepare_source", fail)
    root = tmp_path / "bundle"
    with pytest.raises(ValidationError, match="fixture failure"):
        prep.prepare_arena(native, environment=root, python=sys.executable, allow_network=True)
    assert json.loads((root / "failure.json").read_text())["status"] == "failed"
    assert not (root / "arena.json").exists()
    assert not (root / "receipt.json").exists()


def test_install_subprocess_does_not_inherit_target_or_pythonpath(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/unrelated")
    monkeypatch.setenv("PIP_TARGET", "/important")
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-not-needed")
    env = prep._install_env()
    assert "PYTHONPATH" not in env
    assert "PIP_TARGET" not in env
    assert "OPENAI_API_KEY" not in env
    assert env["PYTHONNOUSERSITE"] == "1"


def test_cli_preparation_routes_without_running_cases(monkeypatch, tmp_path, capsys):
    from agentinputlab.cli.main import main
    seen = []
    monkeypatch.setattr(prep, "prepare_arena", lambda *a, **kw: seen.append((a, kw)) or {"status": "installed"})
    assert main(["config", "prepare-arena", "arena.json", "--environment", str(tmp_path / "env"),
                 "--python", sys.executable, "--allow-network"]) == 0
    assert seen[0][0] == ("arena.json",)
    assert seen[0][1]["allow_network"] is True
    assert json.loads(capsys.readouterr().out)["status"] == "installed"


def test_toolsandbox_preparation_does_not_select_state_source(native, tmp_path, monkeypatch):
    from dataclasses import replace
    calls, sources, checks = [], [], []
    monkeypatch.setattr(prep, "_probe_python", lambda _: [3, 11, 15])
    monkeypatch.setattr(prep, "_run", lambda command, **kw: calls.append(command) or "")
    monkeypatch.setattr(prep, "_prepare_source", lambda source, destination, **kw: sources.append(kw))
    monkeypatch.setattr(prep, "_verify_environment", lambda root, digest, **kw: checks.append(kw) or {})
    root = tmp_path / "sandbox"
    result = prep.prepare_arena(replace(native, arena="toolsandbox"), environment=root,
                                python=sys.executable, allow_network=True)
    assert result["recipe"] == "toolsandbox-isolated.v1"
    assert sources == checks == [{"arena": "toolsandbox"}]
    installed_args = next(command for command in calls if "install" in command)
    assert str(root / "source") in installed_args
    assert "--no-deps" not in installed_args
    assert "httpx==0.27.2" in installed_args
    assert NativeArenaConfig.from_file(root / "arena.json").arena == "toolsandbox"


@pytest.mark.parametrize("version", [[3, 9, 20], [3, 13, 0]])
def test_toolsandbox_incompatible_python_before_creation(native, tmp_path, monkeypatch, version):
    from dataclasses import replace
    monkeypatch.setattr(prep, "_probe_python", lambda _: version)
    root = tmp_path / "sandbox"
    with pytest.raises(ValidationError, match="3.10.*3.12"):
        prep.prepare_arena(replace(native, arena="toolsandbox"), environment=root,
                           python=sys.executable, allow_network=True)
    assert not root.exists()


@pytest.mark.parametrize("arena", ["tau_banking", "tau_retail"])
def test_tau_preparation_uses_native_source_and_text_dependencies(tmp_path, monkeypatch, arena):
    calls = []
    monkeypatch.setattr(prep, "_probe_python", lambda _: [3, 12, 3])
    monkeypatch.setattr(prep, "_run", lambda command, **kw: calls.append(command) or "")
    monkeypatch.setattr(prep, "_prepare_source", lambda *a, **kw: None)
    monkeypatch.setattr(prep, "_verify_environment", lambda *a, **kw: {})
    native = NativeArenaConfig(arena, str(tmp_path / "source"), environment_backend=BACKEND)
    root = tmp_path / "bundle"
    result = prep.prepare_arena(native, environment=root, python=sys.executable, allow_network=True)
    assert result["model_calls"] == 0
    assert prep._source(root, arena).commit == "cf71a8070269883e38a365ffa85f78f46844c1f4"
    install = next(command for command in calls if "install" in command)
    assert str(root / "source") + "[knowledge]" in install
    assert not any("[all]" in arg or "[voice]" in arg or "torch" in arg for arg in install)
    prepared = NativeArenaConfig.from_file(root / "arena.json")
    assert prepared.arena == arena
    assert prepared.environment_backend == native.environment_backend
    assert prepared.parameters == native.parameters


@pytest.mark.parametrize("version", [[3, 11, 15], [3, 14, 0]])
def test_tau_requires_native_supported_python_before_writes(tmp_path, monkeypatch, version):
    monkeypatch.setattr(prep, "_probe_python", lambda _: version)
    native = NativeArenaConfig("tau_retail", str(tmp_path / "source"), environment_backend=BACKEND)
    with pytest.raises(ValidationError, match="3.12.*3.13"):
        prep.prepare_arena(native, environment=tmp_path / "bundle", python=sys.executable,
                           allow_network=True)
    assert not (tmp_path / "bundle").exists()
