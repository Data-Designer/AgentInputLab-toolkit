from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

import pytest

from agentinputlab.backends import GenerationResponse, Usage
from agentinputlab.contracts import ExternalDependencyError, ValidationError
from agentinputlab.recipes import MCEExample, MCETextFitControls, MCETextOfflineRecipe
from agentinputlab.recipes.mce_text_fit import (
    _context_files,
    _excluded_sdk_imports,
    _TextRuntime,
    validate_mce_text_checkout,
)


class Backend:
    model_id = "fixture"

    def __init__(self, text="<answer>visible answer</answer>"):
        self.requests, self.text = [], text

    def generate(self, request):
        self.requests.append(request)
        return GenerationResponse(text=self.text, model="fixture", usage=Usage(10, 5))


def test_budget_and_seed_are_applied_before_every_actual_attempt():
    backend = Backend()
    runtime = _TextRuntime(backend, MCETextFitControls(max_model_calls=2))
    runtime.generate("system", "one")
    runtime.generate("system", "two")
    assert [r.seed for r in backend.requests] == [17, 18]
    assert all(r.max_tokens == 512 for r in backend.requests)
    with pytest.raises(ValidationError, match="before dispatch"):
        runtime.generate("system", "third")
    assert len(backend.requests) == 2
    assert runtime.input_tokens == 20 and runtime.output_tokens == 10


def test_validation_reference_never_enters_model_prompt():
    backend = Backend()
    runtime = _TextRuntime(backend, MCETextFitControls())
    sample = types.SimpleNamespace(
        question="visible question",
        ground_truth="PRIVATE_REFERENCE",
        to_dict=lambda: {"question": "visible question"},
    )
    result = asyncio.run(runtime.batch_evaluate({}, [sample], "fixture"))
    assert result["summary"]["total"] == 1
    assert "PRIVATE_REFERENCE" not in backend.requests[0].prompt
    assert "PRIVATE_REFERENCE" not in backend.requests[0].system


def test_excluded_sdk_sentinels_raise_and_modules_are_restored():
    before = sys.modules.get("claude_agent_sdk")
    with _excluded_sdk_imports():
        with pytest.raises(ExternalDependencyError, match="excluded SDK"):
            sys.modules["claude_agent_sdk"].query()
        with pytest.raises(ExternalDependencyError, match="excluded SDK"):
            sys.modules["langchain_openai"].OpenAIEmbeddings()
    assert sys.modules.get("claude_agent_sdk") is before


def test_context_never_loads_generated_code_or_symlinks(tmp_path):
    folder = tmp_path / "context"
    folder.mkdir()
    (folder / "note.md").write_text("Plain guidance")
    assert _context_files(tmp_path) == {"context/note.md": "Plain guidance"}
    (folder / "code.py").write_text("raise RuntimeError('not executed')")
    with pytest.raises(ValidationError, match="non-text"):
        _context_files(tmp_path)
    (folder / "code.py").unlink()
    (folder / "link.md").symlink_to(folder / "note.md")
    with pytest.raises(ValidationError, match="non-text"):
        _context_files(tmp_path)


def test_training_split_overlap_rejected_before_import_or_inference(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    train = MCEExample("same", {"question": "q", "context": "c"}, "a", "train")
    valid = MCEExample("same", {"question": "v", "context": "c"}, "a", "validation")
    backend = Backend()
    with pytest.raises(ValidationError, match="IDs overlap"):
        MCETextOfflineRecipe().fit(
            checkout=Path("not-accessed"),
            output_root=tmp_path / "not-created",
            train=[train],
            validation=[valid],
            backend=backend,
        )
    assert not backend.requests
    assert not (tmp_path / "not-created").exists()


@pytest.mark.parametrize(
    "field,value",
    [("iterations", 0), ("seed", -1), ("max_model_calls", True), ("timeout_seconds", 0)],
)
def test_invalid_controls(field, value):
    with pytest.raises(ValidationError):
        MCETextFitControls(**{field: value})


def test_untracked_source_module_rejected_before_import(monkeypatch, tmp_path):
    from agentinputlab.recipes import MCE_UPSTREAM_COMMIT

    def git(command, **kwargs):
        if "rev-parse" in command:
            return MCE_UPSTREAM_COMMIT
        if "ls-files" in command:
            return "mce/shadow.py\n"
        return ""

    monkeypatch.setattr("agentinputlab.recipes.mce_text_fit.subprocess.check_output", git)
    with pytest.raises(ExternalDependencyError, match="untracked importable"):
        validate_mce_text_checkout(tmp_path)
