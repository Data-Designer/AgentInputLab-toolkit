"""Explicit source-completion fixtures, never a neural fit certificate."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agentinputlab.contracts import ExternalDependencyError, ValidationError
from agentinputlab.recipes import (
    prepare_smartsearch_native_data_source,
    verify_smartsearch_native_data_source,
)
from agentinputlab.recipes import smartsearch_native_data as data
from agentinputlab.recipes import smartsearch_policy as policy
from agentinputlab.recipes.smartsearch_policy import _source_import


@pytest.fixture
def source_fixture(tmp_path, monkeypatch):
    native, official = tmp_path / "native", tmp_path / "official"
    native.mkdir()
    official.mkdir()
    rows = {name: ('"""Source-only fixture ' + name + '."""\n').encode() for name in data._DATA_BLOBS}
    monkeypatch.setattr(data, "_DATA_BLOBS", {n: data._blob(v) for n, v in rows.items()})
    monkeypatch.setattr(data, "_native_files", lambda source: (Path(source).resolve(), {"fixture.py": "a" * 64}))

    def git(root, *args):
        assert Path(root) == official and args[0] == "show"
        assert args[1].startswith(data.SMARTSEARCH_LLAMAFACTORY_COMMIT + ":src/llamafactory/data/")
        return rows[args[1].split(":src/llamafactory/data/", 1)[1]]

    monkeypatch.setattr(data, "_git", git)
    return native, official, rows


def prepare(fixture, tmp_path):
    native, official, _ = fixture
    root = tmp_path / "supplement"
    binding = prepare_smartsearch_native_data_source(checkout=native, official_checkout=official, output_root=root)
    return root, binding


def test_all_seventeen_data_files_create_only_and_source_untouched(source_fixture, tmp_path):
    root, binding = prepare(source_fixture, tmp_path)
    native, official, rows = source_fixture
    assert binding["supplementary_data_python_files"] == len(rows) == 17
    assert not binding["existing_source_modified"]
    assert list(native.iterdir()) == list(official.iterdir()) == []
    assert verify_smartsearch_native_data_source(native, root) == binding
    for name, original in rows.items():
        assert (root / "data" / name).read_bytes() == original
    with pytest.raises(ValidationError, match="create-only"):
        prepare(source_fixture, tmp_path)


@pytest.mark.parametrize("change", ["bytes", "extra_file", "manifest", "symlink"])
def test_corruption_is_rejected_before_import(source_fixture, tmp_path, change):
    root, _ = prepare(source_fixture, tmp_path)
    if change == "bytes":
        (root / "data/template.py").write_text("changed fixture")
    elif change == "extra_file":
        (root / "data/arbitrary.py").write_text("unexpected fixture")
    elif change == "manifest":
        manifest = json.loads((root / "manifest.json").read_text())
        manifest["official_commit"] = "0" * 40
        (root / "manifest.json").write_text(json.dumps(manifest))
    else:
        (root / "unexpected-link").symlink_to(root / "data")
    with pytest.raises(ExternalDependencyError):
        verify_smartsearch_native_data_source(source_fixture[0], root)


def test_wrong_official_objects_do_not_create_partial_mount(source_fixture, tmp_path, monkeypatch):
    monkeypatch.setattr(data, "_git", lambda *a: b"wrong official fixture")
    with pytest.raises(ExternalDependencyError, match="Git object differs"):
        prepare(source_fixture, tmp_path)
    assert not (tmp_path / "supplement").exists()


def test_source_overlap_rejected(source_fixture):
    native, official, _ = source_fixture
    with pytest.raises(ValidationError, match="overlap"):
        prepare_smartsearch_native_data_source(checkout=native, official_checkout=official, output_root=native / "nested")


def test_fresh_child_rechecks_exact_binding_before_any_import(tmp_path):
    bound = {"root": str(tmp_path / "supplement"), "artifact_id": "a" * 64}
    with patch("agentinputlab.recipes.smartsearch_policy.verify_smartsearch_native_data_source", return_value=dict(bound, artifact_id="b" * 64)), patch(
        "agentinputlab.recipes.smartsearch_policy.importlib.import_module") as importer:
        with pytest.raises(ExternalDependencyError, match="launch binding"):
            _source_import(tmp_path, bound)
        importer.assert_not_called()


def test_mount_does_not_replace_general_native_module_path(tmp_path, monkeypatch):
    monkeypatch.delitem(sys.modules, "llamafactory", raising=False)
    monkeypatch.delitem(sys.modules, "llamafactory.data", raising=False)
    bound = {"root": str(tmp_path / "supplement"), "artifact_id": "a" * 64}
    package = SimpleNamespace(__file__=str(tmp_path / "LLaMA-Factory/src/llamafactory/__init__.py"), __path__=["original-only"])
    trainer = SimpleNamespace(__file__=str(tmp_path / "LLaMA-Factory/src/llamafactory/train/tuner.py"))
    spec = SimpleNamespace(name="llamafactory.data", loader=SimpleNamespace(exec_module=lambda m: None))
    module = SimpleNamespace()
    previous_path = list(sys.path)
    with patch.object(policy, "verify_smartsearch_native_data_source", return_value=bound), patch.object(
        policy.importlib.util, "spec_from_file_location", return_value=spec) as factory, patch.object(
        policy.importlib.util, "module_from_spec", return_value=module), patch.object(
        policy.importlib, "import_module", side_effect=[package, trainer]):
        assert _source_import(tmp_path, bound) is trainer
    assert package.__path__ == ["original-only"]
    assert sys.path == previous_path
    assert factory.call_args[0][0] == "llamafactory.data"
    assert sys.modules["llamafactory.data"] is module
    monkeypatch.delitem(sys.modules, "llamafactory.data")
