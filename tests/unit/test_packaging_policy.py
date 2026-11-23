"""Public distribution tests; no paper archive is a required package input."""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_minimal_import_is_dependency_light():
    code = "import agentinputlab,sys; assert not {'torch','transformers','llmlingua','datasets','vllm'} & set(sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_smartsearch_grpo_install_extra_is_native_pinned_and_separate():
    # This extra targets Linux/Python3.11. Read its simple string array without
    # adding a TOML parser dependency to the Python3.10 core test environment.
    project = (ROOT / "pyproject.toml").read_text()
    body = re.search(r"^smartsearch-grpo = \[(.*?)^\]", project, re.MULTILINE | re.DOTALL)
    assert body
    packages = set(re.findall(r'"([^"\n]+)"', body.group(1)))
    assert {"torch==2.6.0", "transformers==4.51.1", "vllm==0.8.4",
            "ray[default]==2.47.1", "tensordict==0.6.2", "pyarrow==21.0.0",
            "antlr4-python3-runtime==4.9.3", "langid==1.1.6"} <= packages
    assert all("==" in package for package in packages)
    assert not any(package.startswith(("flash-attn", "flash_attn", "deepspeed")) for package in packages)
    assert "dependencies = []" in project


def test_smartsearch_grpo_install_documents_separate_native_build():
    instructions = (ROOT / "docs/smartsearch-training.md").read_text()
    assert "agentinputlab[smartsearch-grpo]" in instructions
    assert "FLASH_ATTENTION_FORCE_BUILD=TRUE" in instructions
    assert "--no-build-isolation --no-deps" in instructions
    assert "--no-binary=flash-attn flash-attn==2.8.3" in instructions
    assert "Import or kernel checks are not GRPO training" in instructions


def test_publication_is_manual_and_sdk_stays_private():
    assert (ROOT / ".release/PYPI_PUBLISHING_DISABLED").is_file()
    assert json.loads((ROOT / "packages/sdk/package.json").read_text())["private"] is True
    policy = (ROOT / "PUBLISHING.md").read_text()
    assert "Registry publication is manual" in policy
    assert "requires owner authorization after acceptance" in policy
    assert "npm publication is deferred" in policy
    for path in (ROOT / ".github/workflows").glob("*.yml"):
        assert "twine upload" not in path.read_text()
        assert "npm publish" not in path.read_text()


def test_public_manifest_does_not_include_research_assets():
    manifest = (ROOT / "MANIFEST.in").read_text()
    assert "prune tests" in manifest
    assert "prune results" in manifest
    assert "*.csv" in manifest and "*.pdf" in manifest
    assert "doc/paper" not in manifest
    assert "MIT License" in (ROOT / "LICENSE").read_text()


def test_inventory_includes_retail_without_expanding_frozen_campaign():
    from agentinputlab.campaign import existing_data_campaign_scope
    from agentinputlab.metrics import metric_definition
    scope = existing_data_campaign_scope()
    assert "tau3.retail" in scope.frozen_inventory
    assert "tau3.retail" not in scope.active_cells
    assert "tooltalk.full" in scope.frozen_inventory
    assert "tooltalk.full" not in scope.active_cells
    # New standalone loaders must not expand a frozen campaign or its assets.
    from agentinputlab.benchmarks import production_benchmark_specs
    from agentinputlab.campaign import existing_campaign_assets
    for cell in ("bright.robotics", "bright.stackoverflow"):
        assert cell in {spec.cell_id for spec in production_benchmark_specs()}
        assert cell in scope.frozen_inventory
        assert cell not in scope.active_cells
        assert not any(asset.startswith(cell) for asset in existing_campaign_assets())
    assert metric_definition("terminal_success").name == "terminal_success"
