"""Online attribution/budget/provenance fixtures, not real ACE outcome evidence."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from agentinputlab.api import create_intervention
from agentinputlab.contracts import ValidationError
from agentinputlab.recipes import ACEOnlineControls, ACEOnlineResult
from agentinputlab.recipes.ace_online import (
    _Journal,
    _RecordingCompletions,
    _tracker_type,
    audit_online_stream,
)
from workers.ace.driver import Controls, LLMConfig, WorkerFailure


def configuration(per_call_output_tokens=512):
    return dict(
        api_provider="openai",
        max_tokens_per_call=per_call_output_tokens,
        json_mode=True,
        base_url="http://127.0.0.1:1/v1",
        api_key_env="FIXTURE_NOT_A_SECRET",
        roles={
            r: dict(
                model="fixture",
                fingerprint=dict(provider="fixture", model="fixture", revision="fixture"),
                input_cost_per_million=0.0,
                output_cost_per_million=0.0,
            )
            for r in ("target", "reflector", "curator")
        },
    )


def example(key="one", question="question"):
    return dict(
        example_id=key,
        input=dict(question=question, context="visible only"),
        expected_output="feedback",
    )


def tracker(tmp_path, controls=None):
    c = controls or ACEOnlineControls()
    journal = _Journal(tmp_path)
    t = _tracker_type()(
        Controls(max_rounds=1, max_model_calls=16, max_tokens=100000, max_cost_usd=0.0, seed=17),
        LLMConfig.from_json(json.dumps(configuration(c.per_call_output_tokens))),
        c,
        {"q1": "one", "q2": "two"},
        journal,
    )
    return t, journal


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", -1),
        ("window_size", 0),
        ("per_call_output_tokens", 4097),
        ("per_call_output_tokens", 0),
        ("per_call_output_tokens", True),
        ("per_call_output_tokens", 1024.5),
        ("per_case_model_calls", False),
        ("per_case_output_tokens", 100),
        ("timeout_seconds", 0),
    ],
)
def test_invalid_controls(field, value):
    with pytest.raises(ValidationError):
        replace(ACEOnlineControls(), **{field: value})


@pytest.mark.parametrize("per_call,total", [(128, 1024), (1024, 4096), (4000, 4096)])
def test_configured_cap_not_fixed_acceptance_cap(tmp_path, per_call, total):
    controls = ACEOnlineControls(per_call_output_tokens=per_call, per_case_output_tokens=total)
    t, journal = tracker(tmp_path, controls)
    t.select("q1")
    calls = total // per_call
    for _ in range(calls):
        t.before_provider_attempt("target")
        t.after_provider_response(
            "target", dict(usage=dict(prompt_tokens=10, completion_tokens=per_call))
        )
    with pytest.raises(WorkerFailure, match="shared case budget"):
        t.before_provider_attempt("reflector")
    assert t.by_case["one"]["model_calls"] == calls
    assert t.by_case["one"]["output_tokens"] == calls * per_call
    journal.close()


def test_configured_online_cap_must_match_deployment_before_any_execution(tmp_path):
    from agentinputlab.recipes import ACEOnlineRecipe

    recipe = ACEOnlineRecipe(ACEOnlineControls(per_call_output_tokens=1024))
    with pytest.raises(ValidationError, match="matching cap"):
        recipe.run_stream(
            [example()],
            checkout="never-open",
            llm_config=configuration(512),
            output_root=tmp_path / "never-written",
            allow_test_feedback=True,
            source_and_asset_acknowledged=True,
        )
    assert not (tmp_path / "never-written").exists()


def test_frozen_stream_order_labels_and_unique_attribution():
    rows, adapted = audit_online_stream([example(), example("two", "second")], ACEOnlineControls())
    assert [r.example_id for r in rows] == ["one", "two"]
    assert rows[0].split == "test" and adapted[0]["target"] == "feedback"
    assert adapted[0]["question"] == "question"
    for bad in (
        [example(), example("two")],
        [example(), example("one", "second")],
        [dict(example(), expected_output=None)],
        [],
    ):
        with pytest.raises(ValidationError):
            audit_online_stream(bad, ACEOnlineControls())


def test_feedback_authority_is_required_before_source_or_model():
    recipe = create_intervention("recipe.ace.online_transductive")
    assert recipe.controls.per_case_model_calls == 8
    with pytest.raises(ValidationError, match="explicit transductive"):
        recipe.run_stream(
            [example()], checkout="absent", llm_config=configuration(), output_root="unused"
        )


def test_repeated_question_with_distinct_context_keeps_native_fields_and_budgets(tmp_path):
    second = example("two")
    second["input"]["context"] = "another source document"
    rows, adapted = audit_online_stream([example(), second], ACEOnlineControls())
    assert adapted[0]["question"] == adapted[1]["question"] == "question"
    assert adapted[0]["context"] == "visible only"
    assert adapted[1]["context"] == "another source document"
    journal = _Journal(tmp_path)
    t = _tracker_type()(
        Controls(max_rounds=1, max_model_calls=16, max_tokens=100000, max_cost_usd=0.0, seed=17),
        LLMConfig.from_json(json.dumps(configuration())),
        ACEOnlineControls(per_case_model_calls=1),
        {
            (r["question"], r["context"]): item.example_id
            for item, r in zip(rows, adapted, strict=True)
        },
        journal,
    )
    with pytest.raises(WorkerFailure, match="ambiguous"):
        t.select("question")
    for row in adapted:
        t.select(row["question"], row["context"])
        t.select(row["question"])  # Native inner component retains its selected sample.
        t.before_provider_attempt("target")
        with pytest.raises(WorkerFailure, match="shared case budget"):
            t.before_provider_attempt("reflector")
    assert t.by_case["one"]["model_calls"] == t.by_case["two"]["model_calls"] == 1
    assert t.case_for(adapted[0]) == "one" and t.case_for(adapted[1]) == "two"
    with pytest.raises(WorkerFailure, match="frozen stream"):
        t.select("question", "not a source context")
    journal.close()


def test_shared_cap_covers_initial_window_and_feedback_not_each_phase(tmp_path):
    t, j = tracker(tmp_path, ACEOnlineControls(per_case_model_calls=2))
    t.select("q1")
    t.phase = "initial_evaluation"
    t.before_provider_attempt("target")
    t.phase = "pre_window_evaluation"
    t.before_provider_attempt("target")
    t.phase = "feedback_adaptation"
    with pytest.raises(WorkerFailure, match="shared case budget"):
        t.before_provider_attempt("reflector")
    assert t.by_case["one"]["model_calls"] == 2 and t.model_calls == 2
    t.select("q2")
    t.before_provider_attempt("target")
    assert t.by_case["two"]["model_calls"] == 1
    j.close()


def test_actual_usage_charged_even_if_component_parsing_never_finishes(tmp_path):
    t, j = tracker(tmp_path)
    t.select("q1")
    t.before_provider_attempt("target")
    t.after_provider_response("target", dict(usage=dict(prompt_tokens=2000, completion_tokens=512)))
    assert t.by_case["one"]["output_tokens"] == 512
    assert t.as_dict()["output_tokens"] == 512 and t.as_dict()["input_tokens"] == 2000
    with pytest.raises(WorkerFailure, match="declared output cap"):
        t.after_provider_response(
            "target", dict(usage=dict(prompt_tokens=3, completion_tokens=513))
        )
    assert t.by_case["one"]["output_tokens"] == 1025  # actual breach remains charged
    j.close()


def test_no_cross_case_or_unattributed_dispatch(tmp_path):
    t, j = tracker(tmp_path)
    with pytest.raises(WorkerFailure, match="source example"):
        t.before_provider_attempt("target")
    with pytest.raises(WorkerFailure, match="frozen stream"):
        t.select("not-source")
    j.close()


def test_provider_seed_and_failure_receipt_no_automatic_repeat(tmp_path):
    t, j = tracker(tmp_path)
    t.select("q1")
    calls = []
    observed = []

    def request(**kwargs):
        observed.append(kwargs)
        raise RuntimeError("fixture provider failure")

    recorder = _RecordingCompletions(SimpleNamespace(create=request), t, "target", calls)
    with pytest.raises(RuntimeError):
        recorder.create(model="fixture", max_tokens=512)
    assert len(observed) == len(calls) == 1 and observed[0]["seed"] == 17
    assert calls[0]["error_type"] == "RuntimeError" and "response" not in calls[0]
    with pytest.raises(ValidationError, match="different provider seed"):
        recorder.create(seed=18)
    assert len(observed) == 1
    j.close()


def test_raw_native_directory_not_a_verified_online_artifact(tmp_path):
    with pytest.raises(ValidationError, match="incomplete"):
        ACEOnlineResult.load(tmp_path)


@pytest.mark.parametrize("repeated_question", [False, True])
@pytest.mark.parametrize("native_processor", [False, True])
@pytest.mark.parametrize("configured_cap", [512, 1024])
def test_fixture_online_observers_preserve_preupdate_order_and_close_costs(
    tmp_path, monkeypatch, repeated_question, native_processor, configured_cap
):
    """A fake algorithm tests instrumentation only; no native/model pass claim."""
    import sys
    import types

    import workers.ace.driver as driver
    from agentinputlab.recipes import ACEOnlineRecipe

    native_module = types.ModuleType("ace_online_protocol_fixture")
    utils = types.ModuleType("utils")

    class Response:
        usage = SimpleNamespace(prompt_tokens=7, completion_tokens=4)

        def model_dump(self, mode):
            return dict(
                usage=dict(prompt_tokens=7, completion_tokens=4),
                choices=[dict(message=dict(content="fixture answer, not model evidence"))],
            )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: Response()))
    )

    class Component:
        def __init__(self):
            self.api_client = client

        def generate(self, *, question, **kwargs):
            self.api_client.chat.completions.create(
                model="fixture",
                max_tokens=configured_cap,
                messages=[dict(role="user", content=question)],
            )
            return (
                "fixture answer, not model evidence",
                [],
                dict(model="fixture", prompt_num_tokens=7, response_num_tokens=4),
            )

        def reflect(self, *, question, **kwargs):
            return self.generate(question=question)

        def curate(self, **kwargs):
            return self.generate(question=kwargs["question_context"])

    def native_sample(args_tuple, processor):
        _, task, generator, playbook, *_ = args_tuple
        answer, _, _ = generator.generate(question=task["question"], playbook=playbook)
        return dict(final_answer=answer, success=True), None

    utils.evaluate_single_test_sample = native_sample

    def native_evaluate(processor, generator, playbook, samples, *args, **kwargs):
        for i, item in enumerate(samples):
            utils.evaluate_single_test_sample((i, item, generator, playbook), processor)
        return dict(total=len(samples), accuracy=0), dict(errors=[])

    native_module.evaluate_test_set = native_evaluate

    class FakeACE:
        def __init__(self, **kwargs):
            assert kwargs["max_tokens"] == configured_cap
            self.generator, self.reflector, self.curator = Component(), Component(), Component()
            self.playbook = "fixture-initial-playbook"

        def _run_test(self, **kwargs):
            return None

        def _train_single_sample(self, *, task_dict, **kwargs):
            q = task_dict["question"]
            self.generator.generate(question=q)
            self.reflector.reflect(question=q)
            self.curator.curate(question_context=q)
            self.generator.generate(question=q)
            self.playbook += "|updated:" + q
            return "fixture-pre", "fixture-post", dict(fixture=True)

        def run(self, *, mode, test_samples, data_processor, **kwargs):
            assert mode == "online"
            native_module.evaluate_test_set(
                data_processor, self.generator, self.playbook, test_samples
            )
            for item in test_samples:
                native_module.evaluate_test_set(
                    data_processor, self.generator, self.playbook, [item]
                )
                self._train_single_sample(task_dict=item)
            return dict(fixture=True)

    FakeACE.__module__ = native_module.__name__

    class FakeRuntime:
        def __init__(self, *args, **kwargs):
            assert kwargs.get("data_processor") is processor
            self.ace_class = FakeACE
            self.policy = SimpleNamespace(
                commit="fixture-not-a-real-source-claim",
                source_hashes={"fixture": "not-source-evidence"},
            )

        def _bind_openai_compatible_clients(self, system, *, seed):
            assert seed == 17

    monkeypatch.setattr(driver, "ACEWorkerRuntime", FakeRuntime)
    monkeypatch.setenv("FIXTURE_NOT_A_SECRET", "fixture-not-a-real-credential")
    monkeypatch.setitem(sys.modules, native_module.__name__, native_module)
    monkeypatch.setitem(sys.modules, "utils", utils)
    examples = [example(), example("two", "question" if repeated_question else "second")]
    if repeated_question:
        examples[1]["input"]["context"] = "another source document"

    class NativeProcessor:
        def fingerprint(self):
            return dict(native_metric="fixture_rank_metric_not_QA")

        def validate_online_examples(self, rows):
            assert [r["example_id"] for r in rows] == ["one", "two"]

        def answer_is_correct(self, *args):
            return True

        def evaluate_accuracy(self, *args):
            return 0.25

    processor = NativeProcessor() if native_processor else None
    original_sample = utils.evaluate_single_test_sample

    def observed_processor(args, value):
        if native_processor:
            assert value is processor
        return original_sample(args, value)

    utils.evaluate_single_test_sample = observed_processor
    result = ACEOnlineRecipe(ACEOnlineControls(per_call_output_tokens=configured_cap)).run_stream(
        examples,
        checkout="fixture",
        llm_config=configuration(configured_cap),
        output_root=tmp_path / "run",
        allow_test_feedback=True,
        source_and_asset_acknowledged=True,
        data_processor=processor,
    )
    plan = json.loads((result.root / "plan.json").read_text())
    assert plan["controls"]["per_call_output_tokens"] == configured_cap
    calls = json.loads((result.root / "provider-calls.json").read_text())
    assert all(call["request"]["max_tokens"] == configured_cap for call in calls)
    assert plan.get("native_task_processor") == (processor.fingerprint() if processor else None)
    terminal = result.terminal
    assert terminal["status"] == "native_online_returned", terminal["error"]
    assert len(terminal["windows"]) == 3 and len(terminal["updates"]) == 2
    assert [r["status"] for r in terminal["primary_outcomes"]] == ["native_prediction_returned"] * 2
    assert terminal["usage"]["model_calls"] == 12
    assert all(r["model_calls"] == 6 for r in terminal["per_case_usage"].values())
    assert (
        terminal["windows"][2]["before_playbook_sha256"]
        == terminal["updates"][0]["after_playbook_sha256"]
    )
    assert (
        terminal["windows"][1]["before_playbook_sha256"]
        == terminal["updates"][0]["before_playbook_sha256"]
    )
    (result.root / "provider-calls.json").write_text("[]")
    with pytest.raises(ValidationError, match="hashes differ"):
        ACEOnlineResult.load(result.root)


def test_online_task_binding_is_validated_before_files_or_native_execution(tmp_path):
    from agentinputlab.recipes import ACEOnlineRecipe

    for invalid in (object(), SimpleNamespace(answer_is_correct=lambda *a: True)):
        with pytest.raises(ValidationError, match="processor lacks"):
            ACEOnlineRecipe().run_stream(
                [example()],
                checkout="never-open",
                llm_config=configuration(),
                output_root=tmp_path / "never-written",
                allow_test_feedback=True,
                source_and_asset_acknowledged=True,
                data_processor=invalid,
            )
    assert not (tmp_path / "never-written").exists()
