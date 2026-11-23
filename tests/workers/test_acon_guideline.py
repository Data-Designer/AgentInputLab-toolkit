"""Offline source/transport guard regressions, not real-model acceptance."""

import hashlib
import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from agentinputlab.backends import GenerationResponse, Usage
from agentinputlab.contracts import ExternalDependencyError, ValidationError
from workers.acon.guideline import (
    ACONGuidelineControls,
    _MeteredLLM,
    validate_training_bundle,
)


def bundle(tmp_path, *, excluded=()):
    root = tmp_path / "input"
    root.mkdir()
    hashes = {}
    for role in ("baseline", "optimized"):
        d = root / role / "samples" / "native-case-a"
        d.mkdir(parents=True)
        p = d / "llm_history.json"
        p.write_text(json.dumps([{"role": "user", "content": "synthetic unit fixture"}]))
        hashes[p.relative_to(root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
        p = root / (role + "-evaluations.json")
        p.write_text(json.dumps({"individual": {"native-case-a": {"f1": 0.0}}}))
        hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            dict(
                schema="agentinputlab.acon.training-input.v1",
                split="train",
                benchmark="smolagents",
                train_ids=["native-case-a"],
                excluded_evaluation_ids=list(excluded),
                files=hashes,
            )
        )
    )
    return root, hashlib.sha256(manifest.read_bytes()).hexdigest()


def test_controls_validate_caps_and_types():
    with pytest.raises(ValidationError):
        ACONGuidelineControls(seed=True)
    with pytest.raises(ValidationError):
        ACONGuidelineControls(per_call_output_tokens=513, max_output_tokens=512)


def test_train_bundle_validates_file_identity_and_exclusion(tmp_path):
    root, sha = bundle(tmp_path)
    assert validate_training_bundle(root, sha)["train_ids"] == ["native-case-a"]
    (root / "optimized/samples/native-case-a/llm_history.json").write_text("modified")
    with pytest.raises(ValidationError, match="hash differs"):
        validate_training_bundle(root, sha)


def test_train_bundle_rejects_test_overlap_before_model_calls(tmp_path):
    root, sha = bundle(tmp_path, excluded=["native-case-a"])
    with pytest.raises(ValidationError, match="overlap"):
        validate_training_bundle(root, sha)


def test_unpinned_history_is_not_silently_used(tmp_path):
    root, sha = bundle(tmp_path)
    (root / "baseline/samples/native-case-a/extra.json").write_text("{}")
    with pytest.raises(ValidationError, match="unpinned"):
        validate_training_bundle(root, sha)


def test_missing_case_history_cannot_look_like_no_regression(tmp_path):
    root, sha = bundle(tmp_path)
    p = root / "optimized/samples/native-case-a/llm_history.json"
    p.unlink()
    with pytest.raises(ValidationError):
        validate_training_bundle(root, sha)


@pytest.mark.parametrize("contents", ["not JSON", "[]", "{}", '[{"role":"user"}]'])
def test_invalid_pinned_history_is_not_a_no_regression_fit(tmp_path, contents):
    root, _ = bundle(tmp_path)
    p = root / "optimized/samples/native-case-a/llm_history.json"
    p.write_text(contents)
    manifest = root / "manifest.json"
    data = json.loads(manifest.read_text())
    data["files"][p.relative_to(root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValidationError, match="history|message"):
        validate_training_bundle(root, hashlib.sha256(manifest.read_bytes()).hexdigest())


def test_meter_caps_native_generation_and_never_retries(tmp_path):
    requests = []

    def generate(request):
        requests.append(request)
        return GenerationResponse(text="candidate", usage=Usage(input_tokens=100, output_tokens=7))

    meter = _MeteredLLM(
        SimpleNamespace(generate=generate), ACONGuidelineControls(max_model_calls=1), tmp_path
    )
    assert meter.generate("native prompt", max_tokens=16384) == "candidate"
    assert requests[0].max_tokens == 512 and requests[0].seed == 17
    with pytest.raises(ValidationError, match="before transport"):
        meter.generate("native prompt")
    assert len(requests) == 1 and meter.output_tokens == 7


def test_meter_preserves_failed_attempt_and_unknown_usage(tmp_path):
    requests = []

    def generate(request):
        requests.append(request)
        raise RuntimeError("genuine failed attempt")

    meter = _MeteredLLM(SimpleNamespace(generate=generate), ACONGuidelineControls(), tmp_path)
    with pytest.raises(RuntimeError):
        meter.generate("native prompt")
    with pytest.raises(ValidationError):
        meter.generate("native prompt")
    assert meter.attempts == len(requests) == 1 and meter.usage_unknown
    assert (tmp_path / "0001-error.json").exists()


def test_meter_rejects_missing_usage_even_for_empty_response(tmp_path):
    meter = _MeteredLLM(
        SimpleNamespace(generate=lambda r: GenerationResponse(text="")),
        ACONGuidelineControls(),
        tmp_path,
    )
    with pytest.raises(ExternalDependencyError, match="token usage"):
        meter.generate("native prompt")
    assert meter.usage_unknown and meter.attempts == 1


def test_public_recipe_api_is_importable_without_native_stack():
    from agentinputlab.recipes import ACONGuidelinePipeline

    assert callable(ACONGuidelinePipeline.fit)


def test_entry_import_failure_has_zero_call_terminal(tmp_path, monkeypatch):
    import workers.acon.guideline as module

    root, sha = bundle(tmp_path)
    checkout = tmp_path / "source"
    (checkout / "prompts").mkdir(parents=True)
    monkeypatch.setattr(module, "validate_guideline_checkout", lambda _: (checkout, {}))
    monkeypatch.setattr(
        module,
        "_load_manifest",
        lambda _: (
            {
                "prompt_profiles": {
                    "smolagents.history.v1": {
                        "mode": "history",
                        "prompt_dir": "prompts",
                        "user_template": "user",
                    }
                }
            },
            "unused",
        ),
    )

    @contextmanager
    def import_failure(*args):
        raise ExternalDependencyError("synthetic missing dependency")
        yield  # pragma: no cover

    monkeypatch.setattr(module, "guideline_runtime", import_failure)
    backend = SimpleNamespace(model_id="synthetic", fingerprint=lambda: {"model": "synthetic"})
    out = tmp_path / "fit"
    with pytest.raises(ExternalDependencyError):
        module.ACONGuidelinePipeline().fit(
            checkout=checkout,
            training_root=root,
            training_manifest_sha256=sha,
            output_root=out,
            backend=backend,
        )
    terminal = json.loads((out / "fit-terminal.json").read_text())
    assert terminal["status"] == "entry_failure"
    assert terminal["model_calls"] == 0 and terminal["usage_unknown"] is False
