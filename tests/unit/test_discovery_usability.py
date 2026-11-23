"""Consumer discovery, actionable errors and no-network preflight contracts."""

import json
from unittest.mock import patch

import pytest

from agentinputlab.api import create_intervention, describe_intervention
from agentinputlab.benchmarks import describe_benchmark, list_benchmarks, load_benchmark
from agentinputlab.cli.main import main
from agentinputlab.contracts.errors import RegistryError, ValidationError
from agentinputlab.registry import AlgorithmRegistry


def test_dataset_inventory_is_metadata_only(tmp_path, monkeypatch):
    cache = tmp_path / "must-not-exist"
    monkeypatch.setenv("AGENTINPUTLAB_CACHE", str(cache))
    with patch("agentinputlab.benchmarks.cache.download_artifact_streamed") as download:
        rows = list_benchmarks()
        assert len(rows) == 31
        assert sum(row["automatic_loader"] for row in rows) == 21
        assert len({row["cell_id"] for row in rows}) == 31
        json.dumps(rows)
        download.assert_not_called()
    assert not cache.exists()


def test_dataset_details_and_filters():
    rows = list_benchmarks(search="BEIR")
    assert rows and all("beir" in row["cell_id"] for row in rows)
    assert describe_benchmark("beir.fever")["exceeds_default_download_limit"] is True
    assert describe_benchmark("beir.arguana")["split"] == "test"
    assert describe_benchmark("tau3.banking_knowledge")["runtime_parameters"] == [
        "checkpoints", "evaluation_bridge",
    ]
    native = describe_benchmark("tooltalk.full")
    assert native["automatic_loader"] is False
    assert native["prepare_command"] is None
    assert native["source_download_bytes"] is None
    with pytest.raises(RegistryError, match="did you mean: beir.arguana"):
        describe_benchmark("beir.arguan")


@pytest.mark.parametrize("cell, kwargs", [
    ("beir.fiqa_2018", {"typo": True}),
    ("beir.arguana", {"checkpoints": []}),
    ("musique.answerable", {"evaluation_bridge": object()}),
    ("tau3.banking_knowledge", {}),
    ("tau3.banking_knowledge", {"checkpoints": [], "evaluation_bridge": object(), "typo": 1}),
])
def test_bad_runtime_fails_before_any_preparation(cell, kwargs):
    with patch("agentinputlab.benchmarks.cache.prepare_benchmark") as prepare:
        with pytest.raises(ValidationError):
            load_benchmark(cell, **kwargs)
        prepare.assert_not_called()


def test_constructor_inspection_does_not_construct():
    def factory(*, dependency, count=3):
        raise AssertionError("must not construct")

    with patch.object(AlgorithmRegistry, "resolve_entrypoint", return_value=factory):
        result = describe_intervention("plan.plan_and_solve.ps_prompt")
    assert result["parameters"][0]["required"] is True
    assert result["parameters"][1]["default"] == 3
    json.dumps(result)


def test_missing_and_extra_constructor_arguments_have_actionable_errors():
    with pytest.raises(ValidationError, match="--parameters"):
        create_intervention("plan.plan_and_solve.ps_prompt", unknown_argument=1)
    with pytest.raises(ValidationError, match="--parameters"):
        create_intervention("baseline.no_op", unknown_argument=1)
    def factory(*, dependency):
        raise AssertionError("must not construct")
    with (
        patch.object(AlgorithmRegistry, "resolve_entrypoint", return_value=factory),
        pytest.raises(ValidationError, match="missing a required .*argument"),
    ):
        create_intervention("plan.plan_and_solve.ps_prompt")


def test_profile_errors_preserve_exact_resolution():
    with pytest.raises(RegistryError, match="choose one of"):
        create_intervention("plan.plan_and_solve")
    with pytest.raises(RegistryError, match="did you mean"):
        create_intervention("plan.plan_and_solve.ps_promp")


def test_cli_discovery_and_debug(capsys):
    assert main(["datasets", "list", "--search", "arguana", "--json"]) == 0
    assert [row["cell_id"] for row in json.loads(capsys.readouterr().out)] == ["beir.arguana"]
    assert main(["datasets", "show", "tooltalk.full"]) == 0
    assert json.loads(capsys.readouterr().out)["automatic_loader"] is False
    assert main(["list", "--releases", "--search", "plan_and_solve", "--json"]) == 0
    assert all("plan_and_solve" in row["id"] for row in json.loads(capsys.readouterr().out))
    assert main(["show", "plan.plan_and_solve.ps_prompt", "--parameters"]) == 0
    assert "parameters" in json.loads(capsys.readouterr().out)
    assert main(["datasets", "show", "does-not-exist"]) == 2
    assert "datasets list" in capsys.readouterr().err
    with pytest.raises(RegistryError):
        main(["--debug", "datasets", "show", "does-not-exist"])


def test_all_registered_constructor_descriptions_are_serializable():
    for _, release in AlgorithmRegistry.default().list_releases():
        result = describe_intervention(release.id)
        assert result["id"] == release.id
        assert isinstance(result["parameters"], list)
        json.dumps(result)
