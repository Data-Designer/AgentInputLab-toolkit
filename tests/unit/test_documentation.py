"""Guides are available without optional runtimes or repository access."""

import json
from pathlib import Path

import pytest

from agentinputlab.cli.main import main
from agentinputlab.contracts.errors import RegistryError
from agentinputlab.documentation import list_guides, read_guide


def test_single_source_guides():
    root = Path(__file__).resolve().parents[2] / "docs"
    expected = sorted(path.relative_to(root).as_posix()[:-3] for path in root.rglob("*.md"))
    assert sorted(list_guides()) == expected
    for topic in expected:
        assert read_guide(topic) == (root / (topic + ".md")).read_text(encoding="utf-8")


@pytest.mark.parametrize("topic", ["../README", "/etc/passwd", "configuration.md", "missing", ""])
def test_reject_arbitrary_paths(topic):
    with pytest.raises(RegistryError, match="unknown documentation topic"):
        read_guide(topic)


def test_docs_cli(capsys):
    assert main(["docs", "--list", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["topics"] == list_guides()
    assert main(["docs", "configuration", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"topic": "configuration", "offline": True, "content": read_guide("configuration")}
    assert main(["docs"]) == 0
    assert capsys.readouterr().out == read_guide("getting-started")
    assert main(["docs", "../README"]) == 2
    assert "Traceback" not in capsys.readouterr().err
