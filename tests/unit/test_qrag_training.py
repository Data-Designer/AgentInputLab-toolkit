"""Schema/API fixtures; these do not execute native policy training."""

import json
from dataclasses import replace

import pytest

from agentinputlab.api import create_intervention
from agentinputlab.contracts import ValidationError
from agentinputlab.recipes import QRAGFitControls, QRAGValuePolicyTraining
from agentinputlab.recipes.qrag import (
    _counted_embedding_forward,
    _NativeCheckpointStore,
    _NativeRecords,
    _sha,
    audit_qrag_training_data,
)


def data(tmp_path, task="hotpotqa"):
    files = []
    for split in ("train", "dev"):
        row = dict(
            _id=split,
            id=split,
            question=f"{split} fixture question?",
            answer="fixture answer",
            context=[["a", ["a fact"]], ["b", ["b fact"]]],
            supporting_facts=[["a", 0]],
            paragraphs=[dict(title="", paragraph_text="source paragraph")],
            question_decomposition=[dict(paragraph_support_idx=0)],
            chunks=["source chunk"],
            references_idx=[0],
        )
        p = tmp_path / f"{split}.json"
        p.write_text(json.dumps([row]))
        files.append(p)
    return files


def test_registry_and_native_optimizer_plan(tmp_path):
    method = create_intervention("training.qrag.value_policy")
    assert isinstance(method, QRAGValuePolicyTraining)
    c = QRAGFitControls(steps_count=2, batch_size=3, environments=2, accumulate_grads=2)
    train, dev = data(tmp_path)
    audit = audit_qrag_training_data(train, dev, c)
    assert audit["native_updates"] == 4 and audit["optimizer_steps"] == 2
    assert audit["transitions_per_rollout"] == 4
    assert audit["no_dummy_or_fabricated_reward_labels"]


@pytest.mark.parametrize("task", ["hotpotqa", "2WikiMultihopQA", "musique", "babilong"])
def test_native_labeled_tasks_preserve_caller_records(tmp_path, task):
    train, dev = data(tmp_path)
    before = train.read_bytes()
    audit = audit_qrag_training_data(train, dev, QRAGFitControls(task=task))
    assert audit["task"] == task and audit["files"]["train"]["rows"] == 1
    records = _NativeRecords(train, task, _sha(train))
    assert records.name() == task and len(records) == 1 and records[0] == json.loads(before)[0]
    assert train.read_bytes() == before


@pytest.mark.parametrize(
    "name,value",
    [
        ("steps_count", 0),
        ("batch_size", True),
        ("accumulate_grads", 0),
        ("seed", -1),
        ("device", "cuda:6"),
        ("task", "unsupported"),
        ("min_free_output_bytes", 0),
        ("warmup_steps", -1),
        ("max_optimizer_steps", 1),
    ],
)
def test_invalid_controls(name, value):
    with pytest.raises(ValidationError):
        replace(QRAGFitControls(), **{name: value})


@pytest.mark.parametrize(
    "change",
    [
        "no_support",
        "unknown_title",
        "out_of_range",
        "duplicate_query",
        "duplicate_id",
        "empty_answer",
        "unhashable_title",
    ],
)
def test_malformed_labels_and_leakage_rejected(tmp_path, change):
    train, dev = data(tmp_path)
    row = json.loads(train.read_text())[0]
    if change == "no_support":
        row["supporting_facts"] = []
    elif change == "unknown_title":
        row["supporting_facts"] = [["absent", 0]]
    elif change == "out_of_range":
        row["supporting_facts"] = [["a", 9]]
    elif change == "duplicate_query":
        row["question"] = "dev fixture question?"
    elif change == "duplicate_id":
        row["_id"] = "dev"
    elif change == "empty_answer":
        row["answer"] = ""
    else:
        row["supporting_facts"] = [[[], 0]]
    train.write_text(json.dumps([row]))
    with pytest.raises(ValidationError):
        audit_qrag_training_data(train, dev, QRAGFitControls())


def test_changed_input_and_byte_cap_are_not_silently_accepted(tmp_path):
    train, dev = data(tmp_path)
    before = _sha(train)
    train.write_text(train.read_text() + " ")
    with pytest.raises(ValidationError, match="differs"):
        _NativeRecords(train, "hotpotqa", before)
    with pytest.raises(ValidationError, match="byte cap"):
        audit_qrag_training_data(train, dev, QRAGFitControls(max_data_bytes=1))
    link = tmp_path / "alias.json"
    link.symlink_to(train)
    with pytest.raises(ValidationError, match="non-symlink"):
        audit_qrag_training_data(link, dev, QRAGFitControls())


def test_fit_requires_explicit_assets_before_optional_runtime(tmp_path):
    with pytest.raises(ValidationError, match="acknowledgement"):
        QRAGValuePolicyTraining().fit(
            checkout=tmp_path,
            initial_checkpoint=tmp_path,
            initial_manifest=tmp_path / "missing.json",
            initial_manifest_sha256="0" * 64,
            train_file=tmp_path / "train.json",
            dev_file=tmp_path / "dev.json",
            output_root=tmp_path / "out",
        )


def test_checkpoint_storage_preserves_best_when_last_changes(tmp_path):
    from types import SimpleNamespace

    calls, events = [], []

    def native_save(agent, path, verbose=False):
        path = __import__("pathlib").Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(agent._update_step))
        calls.append(agent._update_step)

    agent = SimpleNamespace(_update_step=1)
    store = _NativeCheckpointStore(tmp_path, native_save, events)
    store(agent, tmp_path / "last.pt")
    store(agent, tmp_path / "best.pt")
    agent._update_step = 2
    store(agent, tmp_path / "last.pt")
    store(agent, tmp_path / "final.pt")
    assert calls == [1, 2] and len(events) == 4
    assert (tmp_path / "best.pt").read_text() == "1"
    assert (tmp_path / "last.pt").read_text() == (tmp_path / "final.pt").read_text() == "2"
    assert (tmp_path / "last.pt").stat().st_ino == (tmp_path / "final.pt").stat().st_ino
    with pytest.raises(ValidationError, match="escapes"):
        store(agent, tmp_path.parent / "outside.pt")


def test_direct_forward_measurement_preserves_result_identity():
    counters = dict(embedding_forward_attempts=0, embedding_forward_responses=0)
    output = object()

    def native(module, *args, **kwargs):
        assert args == ("input",) and kwargs == {"mask": "native"}
        return output

    wrapper = _counted_embedding_forward(native, counters)
    assert wrapper(object(), "input", mask="native") is output
    assert counters == dict(embedding_forward_attempts=1, embedding_forward_responses=1)


def test_direct_forward_failure_retains_attempt_without_response():
    counters = dict(embedding_forward_attempts=0, embedding_forward_responses=0)

    def native(module):
        raise RuntimeError("native failure fixture")

    with pytest.raises(RuntimeError, match="native failure"):
        _counted_embedding_forward(native, counters)(object())
    assert counters == dict(embedding_forward_attempts=1, embedding_forward_responses=0)
