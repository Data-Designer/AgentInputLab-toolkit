"""Pinned native-source protocol tests with EXPLICIT fixture completions.

No real models, optimizer updates, data downloads, or compatibility
certification. Opt in with a caller-mounted clean source checkout.
"""
import copy
import json
import os
from pathlib import Path

import pytest

from agentinputlab.recipes import (
    SmartSearchCurriculumPreparation,
    SmartSearchProcessTeacher,
    SmartSearchTeacherCompletion,
    SmartSearchTrainingCase,
    verify_smartsearch_preparation_source,
)

SOURCE = os.environ.get("AGENTINPUTLAB_SMARTSEARCH_TRAINING_CHECKOUT")
pytestmark = pytest.mark.skipif(not SOURCE, reason="set pinned SmartSearch training checkout for source-only fixtures")


def test_native_teacher_selection_refinement_sft_dpo_and_prefix(tmp_path):
    pytest.importorskip("requests")
    pytest.importorskip("tqdm")
    source = Path(SOURCE)
    cache = source / "scripts/data_construction/__pycache__"
    before_cache = {p.name: p.read_bytes() for p in cache.glob("*.pyc")}
    verify_smartsearch_preparation_source(source, include_teacher=True)
    train = [SmartSearchTrainingCase("fixture-t1", "Fixture training one", "train"),
             SmartSearchTrainingCase("fixture-t2", "Fixture training two", "train")]
    dev = [SmartSearchTrainingCase("fixture-d", "Fixture development", "validation")]
    demo = "<think>demonstration</think>" * 3
    agent = ("<think>Need a document</think><search>fixture query</search>"
             "<result>result: Fixture document</result><think>Done</think>"
             r"<answer>\boxed{fixture answer}</answer>")
    raw = [{"question": case.question, "golden_answers": ["fixture answer"],
            "output": {"final_response": demo + agent, "pred": "fixture answer",
                       "metric_score": {"f1": float(index == 0)}}}
           for index, case in enumerate(train)]
    calls = []

    def fixture_completion(prompt, **kwargs):
        calls.append((prompt, kwargs))
        if kwargs["stop"]:
            text = "<search>fixture replacement"
        else:
            score = "1" if "Fixture training one" in prompt else "0"
            text = f"<answer>{score}</answer><explanation>fixture rationale</explanation>"
        return SmartSearchTeacherCompletion(text, 100, 20, "a" * 64)

    provenance = {"fixture_only": True, "real_model_calls": 0, "not_real_acceptance": True}
    teacher = SmartSearchProcessTeacher(fixture_completion, model_fingerprint_sha256="a" * 64)
    result = teacher.annotate(checkout=source, output_root=tmp_path / "teacher", train=train,
                              validation=dev, sampled_rollouts=raw, rollout_provenance=provenance)
    assert len(calls) == 3
    assert "### Golden answer\nfixture answer" in calls[0][0]
    assert result["native_selected_rollouts"] == 2
    initial = json.loads((tmp_path / "teacher/scored-rollouts.json").read_text())
    assert initial[0]["response"] == agent
    assert initial[1]["process_scores"][0]["refined"]["search"] == "fixture replacement"
    regenerated = [copy.deepcopy(initial[0])]
    regenerated[0]["outcome_score"]["f1"] = 0.0
    prepared = SmartSearchCurriculumPreparation().prepare(
        checkout=source, output_root=tmp_path / "prepared", train=train, validation=dev,
        initial_scored_rollouts=initial, regenerated_scored_rollouts=regenerated,
        rollout_provenance=provenance,
    )
    assert (prepared.sft_count, prepared.dpo_count, prepared.prefix_count) == (1, 1, 1)
    prefix = json.loads((tmp_path / "prepared/prefix.json").read_text())[0]
    assert "<search>fixture replacement</search>" in prefix["prefix"]
    assert "<result>" not in prefix["prefix"]
    assert {p.name: p.read_bytes() for p in cache.glob("*.pyc")} == before_cache


def test_native_short_rollout_selection_is_not_fabricated(tmp_path):
    pytest.importorskip("requests")
    pytest.importorskip("tqdm")
    train = [SmartSearchTrainingCase("fixture-t", "Fixture short train", "train")]
    dev = [SmartSearchTrainingCase("fixture-d", "Fixture short dev", "validation")]

    def forbidden_provider(*args, **kwargs):
        raise AssertionError("native fourth-think gate should skip this fixture without a model call")

    result = SmartSearchProcessTeacher(forbidden_provider, model_fingerprint_sha256="a" * 64).annotate(
        checkout=SOURCE, output_root=tmp_path / "short", train=train, validation=dev,
        sampled_rollouts=[{"question": train[0].question, "golden_answers": ["fixture"],
                           "output": {"final_response": "<think>single</think>", "pred": "",
                                      "metric_score": {"f1": 0.0}}}],
        rollout_provenance={"fixture_only": True, "real_model_calls": 0},
    )
    assert result["native_selected_rollouts"] == 0
    assert result["usage"]["model_attempts"] == 0
    assert not result["neural_training_performed"]


def test_native_policy_config_preserves_every_non_execution_option(tmp_path):
    yaml = pytest.importorskip("yaml")
    from agentinputlab.recipes.smartsearch_policy import _CONFIGS, _native_config, _source

    source, binding = _source(SOURCE)
    changed = {"model_name_or_path", "trust_remote_code", "dataset", "dataset_dir", "output_dir",
               "deepspeed", "seed", "data_seed", "overwrite_cache", "overwrite_output_dir",
               "push_to_hub", "report_to", "resume_from_checkpoint", "export_dir"}
    for stage in ("sft", "dpo", "merge"):
        original = yaml.safe_load((source / _CONFIGS[stage][0]).read_text())
        config = _native_config(source, stage, tmp_path / "fixture-model", tmp_path / stage, 17)
        assert {k: v for k, v in original.items() if k not in changed} == {
            k: v for k, v in config.items() if k not in changed}
        assert config["trust_remote_code"] is False
    assert len(binding["policy_configs"]) == 9


def test_native_sampling_loop_and_multilingual_f1_use_fixture_completion(tmp_path):
    pytest.importorskip("langid")
    pytest.importorskip("jieba")
    from agentinputlab.recipes import SmartSearchPolicySampler
    from workers.smartsearch.driver import ModelTurn

    class Model:
        def fingerprint(self):
            return {"fixture_only": True, "real_model_calls": 0}

        def count_prompt_tokens(self, prompt):
            return 100

        def generate(self, prompt, **kwargs):
            assert "fixture answer" not in prompt
            assert kwargs["max_new_tokens"] == 8192 and kwargs["seed"] == 17
            return ModelTurn(r"done</think><answer>\boxed{fixture answer}", "stop", "</answer>", 100, 12)

    class Retriever:
        def fingerprint(self):
            return {"fixture_only": True, "corpus_sha256": "b" * 64, "index_sha256": "c" * 64}

        def search(self, *args, **kwargs):
            raise AssertionError("direct answer fixture must not retrieve")

    result = SmartSearchPolicySampler(Model(), Retriever()).sample(
        checkout=SOURCE, case=SmartSearchTrainingCase("fixture", "Fixture question", "train"),
        golden_answers=["fixture answer"], output_root=tmp_path / "native-sampling",
    )
    row = json.loads((tmp_path / "native-sampling/rollout.json").read_text())
    assert row["output"]["pred"] == "fixture answer"
    assert row["output"]["metric_score"]["f1"] == 1.0
    assert result["usage"]["model_calls"] == 1
    assert result["usage"]["retrieval_calls"] == 0


def test_native_grpo_reward_keeps_formula_and_fails_closed(tmp_path):
    from agentinputlab.contracts import ExternalDependencyError
    from agentinputlab.recipes import SmartSearchNativeGRPOReward, SmartSearchTeacherControls

    calls = []

    def fixture_provider(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return SmartSearchTeacherCompletion("<answer>1", 100, 8, "a" * 64)

    def instance(folder, provider):
        return SmartSearchNativeGRPOReward(provider, checkout=SOURCE,
            references={"Fixture train": ["fixture answer"]}, model_fingerprint_sha256="a" * 64,
            controls=SmartSearchTeacherControls(), output_root=tmp_path / folder)

    demo = "<think>fixture demonstration</think>" * 3
    raw = (demo + "<|im_start|>user\nFixture train<|im_end|>"
           "<think>Need evidence</think><search>fixture query</search>"
           r"<result>result: Fixture document</result><answer>\boxed{fixture answer}</answer>")
    score = instance("good", fixture_provider)
    assert score(None, raw, ["fixture answer"]) == pytest.approx(1.1)
    assert calls[0][1] == {"temperature": .6, "max_new_tokens": 2048,
                           "stop": ("</answer>",), "seed": 17}
    assert "### Golden answer\n['fixture answer']" in calls[0][0]
    assert score.usage["model_responses"] == 1 and score.usage["output_tokens"] == 8
    wrong = raw.replace(r"\boxed{fixture answer}", r"\boxed{wrong}")
    assert score(None, wrong, ["fixture answer"]) == pytest.approx(.2)

    def failing(*args, **kwargs):
        raise ConnectionError("explicit fixture transport failure")

    failed = instance("failed", failing)
    with pytest.raises(ConnectionError):
        failed(None, raw, ["fixture answer"])
    with pytest.raises(ExternalDependencyError):
        failed(None, raw, ["fixture answer"])
    assert failed.usage["model_attempts"] == 1
    assert failed.usage["unknown_output_token_reservations"] == 2048
    assert failed.usage["score_responses"] == 0


def test_native_grpo_parquet_schema_and_configuration(tmp_path):
    import ast
    import re

    pq = pytest.importorskip("pyarrow.parquet")
    yaml = pytest.importorskip("yaml")
    from agentinputlab.contracts.serialization import to_jsonable
    from agentinputlab.recipes import (
        SmartSearchGRPOControls,
        SmartSearchGRPOExample,
        prepare_smartsearch_grpo_data,
        verify_smartsearch_grpo_source,
    )
    from agentinputlab.recipes.smartsearch import SMARTSEARCH_TOOL_SCHEMA
    from agentinputlab.recipes.smartsearch_grpo_data import _native_grpo_config

    source, binding = verify_smartsearch_grpo_source(SOURCE)
    assert len(binding["grpo_source"]) == 12
    assert binding["grpo_source"]["src/verl/utils/checkpoint/fsdp_checkpoint_manager.py"] == (
        "640556d9c0801602818fcfd52aa6d2d066b49bf30f30eb0a81c08efde2495f4a"
    )
    tree = ast.parse((source / "data/prepare_dataset.py").read_text())
    schema = next(n.value for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.value, ast.List)
                  and any(isinstance(t, ast.Name) and t.id == "wikipedia_search_schemas" for t in n.targets))
    assert ast.literal_eval(schema) == to_jsonable(SMARTSEARCH_TOOL_SCHEMA)
    artifact = prepare_smartsearch_grpo_data(checkout=source, output_root=tmp_path / "data",
        train=[SmartSearchGRPOExample(SmartSearchTrainingCase("fixture-t", "Fixture train", "train"), ("gold",))],
        validation=[SmartSearchGRPOExample(SmartSearchTrainingCase("fixture-d", "Fixture dev", "validation"), ("devgold",))],
        data_provenance={"fixture_only": True, "real_model_calls": 0})
    row = pq.read_table(artifact.root / "train.parquet").to_pylist()[0]
    assert set(row) == {"data_source", "question", "ability", "reward_model", "extra_info"}
    assert json.loads(row["extra_info"]["func_schemas"]) == ast.literal_eval(schema)
    controls = SmartSearchGRPOControls(train_batch_size=1, ppo_mini_batch_size=1,
                                       gpu_processes=1, bounded_training_variant=True)
    config = _native_grpo_config(source, model=tmp_path / "model", data=artifact,
        output_root=tmp_path / "run", controls=controls, sandbox_url="http://127.0.0.1:12345/fixture",
        reward_module=tmp_path / "reward.py")
    original = yaml.safe_load((source / "src/verl/trainer/config/ppo_trainer.yaml").read_text())
    script = (source / "scripts/train/train.sh").read_text()
    fields = dict(re.findall(r"^([A-Z_]+)=([^\r\n]+)$", script, re.MULTILINE))
    assert config["critic"] == original["critic"]
    assert config["actor_rollout_ref"]["rollout"]["n"] == int(fields["ROLLOUT_N"]) == 8
    assert config["actor_rollout_ref"]["rollout"]["max_turns"] == int(fields["MAX_TURNS"]) == 5
    assert config["data"]["max_prompt_length"] == int(fields["MAX_PROMPT_LENGTH"]) == 8192
    assert config["data"]["max_response_length"] == int(fields["MAX_RESPONSE_LENGTH"]) == 8192
    assert config["trainer"]["total_epochs"] == int(fields["TOTAL_EPOCHS"]) == 2
    assert config["actor_rollout_ref"]["actor"]["optim"]["lr"] == float(fields["LR"]) == 1e-6
    assert config["actor_rollout_ref"]["actor"]["clip_ratio"] == original["actor_rollout_ref"]["actor"]["clip_ratio"]
