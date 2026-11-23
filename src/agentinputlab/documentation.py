"""Read installed guides without a browser, network access or optional deps."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from agentinputlab.contracts.errors import RegistryError


def _root():
    bundled = files("agentinputlab").joinpath("_bundled_docs")
    if bundled.is_dir():
        return bundled
    # Editable/source checkout: docs/ is the single authoring source.
    checkout = Path(__file__).resolve().parents[2]
    if (checkout / "pyproject.toml").is_file() and (checkout / "docs").is_dir():
        return checkout / "docs"
    raise RegistryError("installed guides are missing; reinstall the complete agentinputlab wheel")


def list_guides() -> list[str]:
    """Return safe topic IDs, including nested release guides."""
    topics = []

    def visit(directory, prefix=""):
        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            if child.is_dir():
                visit(child, prefix + child.name + "/")
            elif child.name.endswith(".md"):
                topics.append(prefix + child.name[:-3])

    visit(_root())
    return topics


def read_guide(topic: str) -> str:
    """Read an exact listed topic; arbitrary filesystem paths are not accepted."""
    if topic not in list_guides():
        raise RegistryError("unknown documentation topic; run 'agentinputlab docs --list'")
    return _root().joinpath(*((topic + ".md").split("/"))).read_text(encoding="utf-8")
