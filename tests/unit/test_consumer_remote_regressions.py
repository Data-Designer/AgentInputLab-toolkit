"""Regression cases reproduced in an isolated Linux PyPI consumer environment."""

import errno
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from agentinputlab.artifacts import ArtifactStore
from agentinputlab.backends.llm import GenerationRequest, OpenAICompatibleBackend
from agentinputlab.cli.main import main
from agentinputlab.contracts import ArtifactError


@pytest.mark.parametrize("usage", [None, {}, {"prompt_tokens": 3}, {"completion_tokens": 2}])
def test_missing_usage_is_not_counted_as_zero_or_retried(usage):
    transport = Mock(return_value={"choices": [{"message": {"content": "answer"}}], "usage": usage})
    backend = OpenAICompatibleBackend(
        model_id="fixture", base_url="http://127.0.0.1:8000/v1",
        deployment_identity={"revision": "protocol-fixture"}, transport=transport,
    )
    with pytest.raises(RuntimeError, match="missing usage is unknown, not zero"):
        backend.generate(GenerationRequest(prompt="test"))
    transport.assert_called_once()


def test_absent_usage_is_not_counted_as_zero():
    transport = Mock(return_value={"choices": [{"message": {"content": "answer"}}]})
    backend = OpenAICompatibleBackend(
        model_id="fixture", base_url="http://127.0.0.1:8000/v1",
        deployment_identity={"revision": "protocol-fixture"}, transport=transport,
    )
    with pytest.raises(RuntimeError, match="missing usage is unknown, not zero"):
        backend.generate(GenerationRequest(prompt="test"))
    transport.assert_called_once()


def test_explicit_zero_usage_remains_valid():
    backend = OpenAICompatibleBackend(
        model_id="fixture", base_url="http://127.0.0.1:8000/v1",
        deployment_identity={"revision": "protocol-fixture"},
        transport=lambda *_: {"choices": [{"message": {"content": ""}}],
                              "usage": {"prompt_tokens": 0, "completion_tokens": 0}},
    )
    response = backend.generate(GenerationRequest(prompt="test"))
    assert response.usage.input_tokens == response.usage.output_tokens == 0


@pytest.mark.parametrize("error", [
    PermissionError(errno.EACCES, "Permission denied"),
    FileNotFoundError(errno.ENOENT, "No such file or directory"),
    NotADirectoryError(errno.ENOTDIR, "Not a directory"),
    OSError(errno.EROFS, "Read-only file system"),
])
def test_unusable_cache_has_actionable_error(tmp_path, error):
    with (
        patch.object(Path, "mkdir", side_effect=error),
        pytest.raises(ArtifactError, match="choose a writable directory"),
    ):
        ArtifactStore(tmp_path / "cache")


def test_cli_cache_permission_error_is_short_and_debug_preserves_trace(tmp_path, capsys):
    argv = ["datasets", "prepare", "beir.arguana", "--cache-dir", str(tmp_path)]
    with patch.object(Path, "mkdir", side_effect=PermissionError(errno.EACCES, "Permission denied")):
        assert main(argv) == 2
        stderr = capsys.readouterr().err
        assert "--cache-dir" in stderr and "Traceback" not in stderr
        with pytest.raises(ArtifactError):
            main(["--debug", *argv])


def test_cache_path_is_a_file_preserves_existing_content(tmp_path):
    target = tmp_path / "cache"
    target.write_text("user content")
    with pytest.raises(ArtifactError, match="choose a writable directory"):
        ArtifactStore(target)
    assert target.read_text() == "user content"
