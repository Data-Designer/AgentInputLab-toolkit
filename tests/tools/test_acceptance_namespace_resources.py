"""Namespace-package resource directories are Traversables, not filesystem strings."""
from pathlib import Path

import acceptance_method_bindings as bindings
import pytest


class NamespaceDirectory:
    def __init__(self, root):
        self.root = root

    def joinpath(self, name):
        return NamespaceDirectory(self.root / name)

    def __truediv__(self, name):
        return self.root / name

    def __str__(self):
        raise AssertionError("Do not turn a namespace Traversable repr into a path")


class Backend:
    def fingerprint(self):
        return {"model": "fixture", "model_revision": "fixture-revision"}


@pytest.mark.parametrize("kind", ["reformer", "infocqr", "ask_or_assume"])
def test_source_binding_resolves_child_files_without_coercing_namespace_directory(monkeypatch, kind):
    source = Path(__file__).resolve().parents[2] / "workers"
    original = bindings.resources.files
    monkeypatch.setattr(bindings.resources, "files", lambda name: (
        NamespaceDirectory(source) if name == "workers" else original(name)
    ))
    kwargs = dict(upstream=Path("/unused"), backend=Backend(), model={
        "id": "fixture", "base_url": "http://localhost:1234/v1",
    })
    if kind == "ask_or_assume":
        _, worker = bindings.bind_ask_or_assume(**kwargs)
    else:
        release = bindings.REFORMER_RELEASES[0] if kind == "reformer" else bindings.EXTERNAL_REWRITERS[1]
        _, worker = bindings.bind_external_rewriter(
            release, dataset_revision="fixture", benchmark="fixture", **kwargs,
        )
    assert worker.command[1] == str(source / kind / "driver.py")
    assert Path(worker.command[1]).is_file()
