from __future__ import annotations

import json
from unittest import SkipTest

try:
    import pytest
except ModuleNotFoundError as error:
    if error.name != "pytest":
        raise
    raise SkipTest("pytest is required for this test module") from error

from agentinputlab.cli.main import main
from agentinputlab.contracts.errors import ValidationError
from agentinputlab.runtime_bootstrap import (
    RUNTIME_BOOTSTRAP_PROTOCOL,
    bootstrap_runtime_sources,
    runtime_source_plan,
)


def test_runtime_source_plan_covers_27_methods_and_six_arenas(tmp_path):
    plan = runtime_source_plan(tmp_path)

    assert len(plan) == 33
    assert sum(source.kind == "method" for source in plan) == 27
    assert sum(source.kind == "arena" for source in plan) == 6
    assert all(source.checkout.is_absolute() for source in plan)
    assert all(len(source.commit) == 40 for source in plan)
    assert {source.source_id for source in plan if source.kind == "arena"} == {
        "tooltalk.full",
        "tau3.banking_knowledge",
        "state_bench.common_support",
        "toolsandbox.common_support",
        "appworld_ul.common_support",
        "ale.external_transfer",
    }


def test_runtime_bootstrap_dry_run_returns_actionable_resolution(tmp_path):
    report = bootstrap_runtime_sources(
        tmp_path,
        method_ids=("rewrite.query2doc",),
        include_arenas=False,
        allow_network=False,
    )

    assert report["protocol"] == RUNTIME_BOOTSTRAP_PROTOCOL
    assert report["source_count"] == 1
    assert report["ready_count"] == 0
    assert report["actionable_count"] == 1
    assert report["sources"][0]["resolution_kind"] == "public_runtime_bootstrap"
    assert "--allow-network" in report["sources"][0]["next_step"]
    assert len(report["plan_sha256"]) == 64


def test_runtime_source_plan_rejects_unknown_method(tmp_path):
    with pytest.raises(ValidationError, match="unknown runtime bootstrap methods"):
        runtime_source_plan(tmp_path, method_ids=("unknown.method",))


def test_campaign_bootstrap_cli_is_non_mutating_without_network(tmp_path, capsys):
    code = main(
        [
            "campaign",
            "bootstrap",
            "--root",
            str(tmp_path),
            "--method",
            "rewrite.query2doc",
            "--no-arenas",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert code == 2
    assert payload["source_count"] == 1
    assert list(tmp_path.iterdir()) == []


def test_tooltalk_source_is_registered_at_provider_revision(tmp_path):
    from workers.official_arenas.providers.tooltalk import ToolTalkProvider

    rows = {row.source_id: row for row in runtime_source_plan(tmp_path)}
    source = rows["tooltalk.full"]
    assert source.repo_url == "https://github.com/microsoft/ToolTalk"
    assert source.commit == ToolTalkProvider.source_revision


def test_arena_only_plan_does_not_select_methods_or_other_arenas(tmp_path):
    plan = runtime_source_plan(tmp_path, method_ids=(), arena_ids=("tooltalk.full",))
    assert [row.source_id for row in plan] == ["tooltalk.full"]
    assert plan[0].checkout == tmp_path / "_arenas/tooltalk.full/checkout"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("arenas", [("unknown",), ("tooltalk.full", "tooltalk.full")])
def test_invalid_arena_selection_is_rejected(tmp_path, arenas):
    with pytest.raises(ValidationError):
        runtime_source_plan(tmp_path, method_ids=(), arena_ids=arenas)


def test_disabled_arenas_cannot_be_selected(tmp_path):
    with pytest.raises(ValidationError):
        runtime_source_plan(tmp_path, arena_ids=("tooltalk.full",), include_arenas=False)


def test_no_sources_is_rejected(tmp_path):
    with pytest.raises(ValidationError):
        runtime_source_plan(tmp_path, method_ids=(), include_arenas=False)


def test_cli_can_select_only_tooltalk_without_network(tmp_path, capsys):
    code = main(["campaign", "bootstrap", "--root", str(tmp_path), "--arena", "tooltalk.full"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert [row["source_id"] for row in payload["sources"]] == ["tooltalk.full"]
    assert list(tmp_path.iterdir()) == []


def test_cli_can_select_method_and_arena(tmp_path, capsys):
    code = main(["campaign", "bootstrap", "--root", str(tmp_path),
                 "--method", "rewrite.query2doc", "--arena", "tooltalk.full"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert {row["source_id"] for row in payload["sources"]} == {"rewrite.query2doc", "tooltalk.full"}
    assert list(tmp_path.iterdir()) == []


def test_three_evaluation_arena_pins_match_native_providers(tmp_path):
    from workers.official_arenas.providers.statebench import StateBenchProvider
    from workers.official_arenas.providers.toolsandbox import ToolSandboxProvider
    from workers.official_arenas.providers.tooltalk import ToolTalkProvider

    expected = {"tooltalk.full": ToolTalkProvider.source_revision,
                "state_bench.common_support": StateBenchProvider.source_revision,
                "toolsandbox.common_support": ToolSandboxProvider.source_revision}
    plan = runtime_source_plan(tmp_path, method_ids=(), arena_ids=tuple(expected))
    assert {row.source_id: row.commit for row in plan} == expected


def test_arena_bootstrap_preserves_unrelated_directory(tmp_path):
    checkout = tmp_path / "_arenas/tooltalk.full/checkout"
    checkout.mkdir(parents=True)
    protected = checkout / "user-data.txt"
    protected.write_text("keep me")
    report = bootstrap_runtime_sources(tmp_path, method_ids=(), arena_ids=("tooltalk.full",),
                                       allow_network=True)
    assert report["ready_count"] == 0
    assert report["source_count"] == 1
    assert protected.read_text() == "keep me"
    assert not (checkout / ".git").exists()


def test_cli_rejects_conflicting_arena_selection(tmp_path, capsys):
    code = main(["campaign", "bootstrap", "--root", str(tmp_path),
                 "--arena", "tooltalk.full", "--no-arenas"])
    assert code != 0
    assert "include_arenas=False" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []
