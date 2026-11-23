"""Offline artifact identity/security tests. Synthetic fixtures, not real fits."""

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

pytest.importorskip(
    "jinja2", reason="install agentinputlab[acon-fit]; exercised separately with the extra"
)

from agentinputlab.contracts import ValidationError
from agentinputlab.contracts.serialization import content_hash
from workers.acon.fitted import (
    ACONGuidelineArtifact,
    validate_fitted_template,
    validate_training_signals,
)

TEXT = "Synthetic task: {{ task }}\nHistory: {{ history }}\nPrevious: {{ prev_summary }}"


def artifact():
    values = dict(
        template=TEXT,
        template_sha256=hashlib.sha256(TEXT.encode()).hexdigest(),
        base_profile="smolagents.history.v1",
        source_hashes={"synthetic.py": "a" * 64},
        backend_fingerprint={"model": "synthetic", "revision": "fixture"},
        training_manifest_sha256="b" * 64,
        train_ids=("synthetic-train",),
        excluded_evaluation_ids=("synthetic-dev",),
        fit_artifact_identity="c" * 64,
        provenance={
            "fit_start_sha256": "d" * 64,
            "fit_terminal_sha256": "e" * 64,
            "analysis_sha256": "f" * 64,
            "candidate_path": "optimized_prompts/synthetic.jinja",
        },
        protocol="agentinputlab.acon.fitted-history.literal-placeholders.v1",
    )
    return ACONGuidelineArtifact(artifact_id=content_hash(values), **values)


def test_artifact_json_round_trip_is_content_addressed(tmp_path):
    value = artifact()
    path = tmp_path / "artifact.json"
    value.save(path)
    loaded = ACONGuidelineArtifact.load(path, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert loaded.as_dict() == value.as_dict()
    with pytest.raises(FileExistsError):
        value.save(path)
    with pytest.raises(ValidationError, match="checksum"):
        ACONGuidelineArtifact.load(path, sha256="0" * 64)


def test_payload_mutation_cannot_keep_an_old_artifact_id():
    value = artifact()
    with pytest.raises(ValidationError, match="payload"):
        replace(value, backend_fingerprint={"model": "changed"})


@pytest.mark.parametrize(
    "attack",
    [
        "{{ cycler.__init__.__globals__.os.system('id') }}",
        "{% include '/etc/passwd' %}",
        "{% for n in range(999999999) %}x{% endfor %}",
        "{{ history|attr('__class__') }}",
        "{% set x = 'payload' %}{{ task }}{{ history }}{{ prev_summary }}",
        "{{ self }}{{ task }}{{ history }}{{ prev_summary }}",
    ],
)
def test_generated_executable_or_indirect_jinja_is_rejected_without_evaluation(attack):
    with pytest.raises(ValidationError):
        validate_fitted_template(attack)


def test_all_source_placeholders_must_be_kept():
    assert validate_fitted_template(TEXT) == ("history", "prev_summary", "task")
    with pytest.raises(ValidationError, match="placeholders"):
        validate_fitted_template("{{ task }} {{ history }}")


def test_cross_model_artifact_use_fails_before_source_or_output_side_effects(tmp_path):
    backend = SimpleNamespace(fingerprint=lambda: {"model": "different"})
    with pytest.raises(ValidationError, match="backend/model"):
        artifact().compress_history(
            checkout="missing",
            backend=backend,
            task="fixture",
            history="fixture",
            output_root=tmp_path / "must-not-exist",
            evaluation_id="synthetic-dev",
        )
    assert not (tmp_path / "must-not-exist").exists()


def test_training_or_unfrozen_evaluation_id_cannot_be_used(tmp_path):
    value = artifact()
    backend = SimpleNamespace(fingerprint=lambda: value.backend_fingerprint)
    for case in ("synthetic-train", "unregistered-test"):
        with pytest.raises(ValidationError, match="excluded before"):
            value.compress_history(
                checkout="missing",
                backend=backend,
                task="fixture",
                history="fixture",
                output_root=tmp_path / "must-not-exist",
                evaluation_id=case,
            )
    assert not (tmp_path / "must-not-exist").exists()


def test_loaded_json_does_not_execute_template_programs(tmp_path):
    values = artifact().as_dict()
    values["template"] = "{{ cycler.__init__ }}"
    values["template_sha256"] = hashlib.sha256(values["template"].encode()).hexdigest()
    path = tmp_path / "untrusted.json"
    path.write_text(json.dumps(values))
    with pytest.raises(ValidationError):
        ACONGuidelineArtifact.load(path, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def test_truncated_analysis_fallback_cannot_be_promoted_as_completed_training():
    with pytest.raises(ValidationError, match="parsed training signals"):
        validate_training_signals(
            [{"raw_output": "synthetic truncated JSON", "task_name": "fixture"}]
        )
    with pytest.raises(ValidationError, match="parsed training signals"):
        validate_training_signals([{"task_name": "fixture"}])
    validate_training_signals([{"root_cause_categories": ["synthetic context-loss fixture"]}])


def test_promotion_validates_all_pinned_metadata_without_running_a_model(tmp_path, monkeypatch):
    import workers.acon.fitted as module
    from tests.workers.test_acon_guideline import bundle

    train, train_sha = bundle(tmp_path)
    source = tmp_path / "source"
    (source / "prompts").mkdir(parents=True)
    (source / "prompts/base.jinja").write_text(TEXT)
    hashes = {"synthetic-source.py": "a" * 64}
    monkeypatch.setattr(module, "validate_guideline_checkout", lambda _: (source, hashes))
    monkeypatch.setattr(
        module,
        "_load_manifest",
        lambda _: (
            {
                "prompt_profiles": {
                    "smolagents.history.v1": {
                        "mode": "history",
                        "prompt_dir": "prompts",
                        "user_template": "base",
                    }
                }
            },
            "unused",
        ),
    )
    backend = SimpleNamespace(
        fingerprint=lambda: {"model": "SYNTHETIC-UNIT-FIXTURE"},
        generate=lambda _: pytest.fail("artifact promotion cannot call a model"),
    )
    root = tmp_path / "fit"
    root.mkdir()
    (root / "optimized_prompts").mkdir()
    candidate = "optimized_prompts/fixture.jinja"
    (root / candidate).write_text(TEXT)
    candidate_sha = hashlib.sha256(TEXT.encode()).hexdigest()
    start = {
        "source_hashes": hashes,
        "training_manifest_sha256": train_sha,
        "base_profile": "smolagents.history.v1",
        "backend": backend.fingerprint(),
        "controls": {"seed": 17},
    }
    terminal = {
        "status": "candidates_generated",
        "source_hashes": hashes,
        "training_manifest_sha256": train_sha,
        "candidates": {candidate: candidate_sha},
        "artifact_identity": content_hash(
            {
                "candidates": {"fixture.jinja": candidate_sha},
                "source": hashes,
                "training": train_sha,
                "backend": backend.fingerprint(),
                "controls": start["controls"],
                "base_profile": start["base_profile"],
            }
        ),
    }
    for name, value in (
        ("fit-start.json", start),
        ("fit-terminal.json", terminal),
        (
            "aggregated_history_regressions.json",
            [{"remediation_recommendations": ["synthetic unit signal"]}],
        ),
    ):
        (root / name).write_text(json.dumps(value))
    args = dict(
        checkout=source,
        fit_root=root,
        training_root=train,
        backend=backend,
        candidate_path=candidate,
        fit_start_sha256=hashlib.sha256((root / "fit-start.json").read_bytes()).hexdigest(),
        fit_terminal_sha256=hashlib.sha256((root / "fit-terminal.json").read_bytes()).hexdigest(),
        analysis_sha256=hashlib.sha256(
            (root / "aggregated_history_regressions.json").read_bytes()
        ).hexdigest(),
    )
    result = ACONGuidelineArtifact.from_fit(**args)
    assert result.template == TEXT and result.train_ids == ("native-case-a",)
    with pytest.raises(ValidationError, match="checksum"):
        ACONGuidelineArtifact.from_fit(**{**args, "analysis_sha256": "0" * 64})
