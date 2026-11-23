"""Consumer-facing commands and fail-fast configuration, without real model calls."""

import contextlib
import io
import json
import os
import subprocess
import sys
import traceback
import urllib.error
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from agentinputlab import __version__
from agentinputlab.backends.llm import (
    GenerationRequest,
    OpenAICompatibleBackend,
    OpenAICompatibleTransportError,
    openai_compatible,
)
from agentinputlab.cli.main import main
from agentinputlab.contracts.errors import ValidationError


def backend(**overrides):
    args = {
        "model_id": "test-model",
        "base_url": "https://example.test/v1",
        "deployment_identity": {"revision": "fixture-revision"},
    }
    args.update(overrides)
    return OpenAICompatibleBackend(**args)


@pytest.mark.parametrize("value", ["", "  ", None, 7, True])
def test_model_id_is_validated_without_transport(value):
    with pytest.raises(ValidationError, match="model_id"):
        backend(model_id=value)


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True, "30", None])
def test_timeout_is_validated_without_transport(value):
    with pytest.raises(ValidationError, match="timeout_seconds"):
        backend(timeout_seconds=value)


@pytest.mark.parametrize("url", [
    None, "", "localhost:8000", "http://[broken", "http://host:invalid/v1",
    "http://host:70000/v1", "http://:8000/v1", " https://host/v1", "http://host/\nv1",
    "http://host/v1/chat/completions", "http://host/v1/chat/completions/",
])
def test_invalid_base_url_is_rejected_before_dispatch(url):
    with pytest.raises(ValidationError, match="base_url"):
        backend(base_url=url)


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:8000/v1/", "http://[::1]:8000/v1", "https://host/custom/api",
])
def test_custom_api_roots_and_ipv6_are_supported(url):
    assert backend(base_url=url).base_url == url.rstrip("/")


@pytest.mark.parametrize("status,hint", [
    (401, "api_key"), (403, "permissions"), (404, "base_url"),
    (429, "concurrency"), (500, "server status"), (302, "server status"),
])
def test_safe_status_and_actionable_hint_without_secret_or_retry(status, hint):
    secret = "unit-test-credential-never-a-real-secret"
    opener = Mock()
    opener.open.side_effect = urllib.error.HTTPError(
        "https://example.test/v1", status, secret, {}, io.BytesIO(secret.encode()),
    )
    with (
        patch.object(openai_compatible.urllib.request, "build_opener", return_value=opener),
        pytest.raises(OpenAICompatibleTransportError) as caught,
    ):
        backend(api_key=secret).generate(GenerationRequest(prompt="fixture"))
    error = caught.value
    assert error.http_status == status
    assert f"HTTP {status}" in str(error) and hint in str(error)
    assert secret not in "".join(traceback.format_exception(type(error), error, error.__traceback__))
    assert error.__cause__ is None and error.__context__ is None
    assert opener.open.call_count == 1


def test_untrusted_status_cannot_become_a_diagnostic():
    def transport(*_args):
        raise openai_compatible._OpenAICompatibleHTTPStatusError("secret")

    with pytest.raises(OpenAICompatibleTransportError) as caught:
        backend(transport=transport).generate(GenerationRequest(prompt="fixture"))
    assert caught.value.http_status is None
    assert str(caught.value) == "OpenAI-compatible transport failed"
    assert caught.value.__context__ is None


def test_version_does_not_require_a_registry_or_subcommand():
    output = io.StringIO()
    with contextlib.redirect_stdout(output), pytest.raises(SystemExit) as caught:
        main(["--registry", "/nonexistent-registry", "--version"])
    assert caught.value.code == 0
    assert output.getvalue().strip() == f"agentinputlab {__version__}"


@pytest.mark.parametrize("arguments", [["--help"], ["--version"], ["smoke", "--json"]])
def test_python_module_entrypoint_outside_checkout(tmp_path, arguments):
    # Locate the package under test, whether installed or imported from src/.
    import agentinputlab

    source = str(Path(agentinputlab.__file__).resolve().parent.parent)
    env = {**os.environ, "PYTHONPATH": source}
    result = subprocess.run(
        [sys.executable, "-m", "agentinputlab", *arguments],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    if arguments[0] == "smoke":
        payload = json.loads(result.stdout)
        assert payload["ok"] and payload["offline"]
