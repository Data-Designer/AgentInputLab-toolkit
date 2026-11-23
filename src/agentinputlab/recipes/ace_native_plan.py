"""Common contracts for ACE's explicitly noninteractive native tool-plan feedback.

Task-specific implementations invoke registered native tools and scorers. This
module neither executes generated Python nor supplies a replacement scorer.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from agentinputlab.api import _state
from agentinputlab.artifacts import load_pinned_json_mapping
from agentinputlab.contracts import ExternalDependencyError, ValidationError
from agentinputlab.contracts.serialization import canonical_json, content_hash, to_jsonable

from .ace import ACEExample, validate_split_isolation

_NATIVE_LOCK = threading.RLock()
PLAN_INSTRUCTION = (
    "Solve the supplied task as one noninteractive tool plan. Preserve all visible "
    "task rules and function schemas. Put exactly one JSON object in your final_answer string: "
    '{"actions":[{"name":"visible_function_name","arguments":{}}],"response":"message to user"}. '
    "Use only visible function names with their required arguments. Actions execute in order "
    "without another model call or access to their returned values while planning. "
    "Do not output Python, code fences, unknown functions, or extra fields.\n"
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(to_jsonable(value), stream, sort_keys=True, indent=2, allow_nan=False)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate tool-plan JSON key")
        result[key] = value
    return result


def parse_native_tool_plan(text, *, tool_names, tool_cap):
    """Decode a complete plan with advertised names, never fences or code repair."""
    if not isinstance(text, str) or type(tool_cap) is not int or tool_cap < 0:
        raise ValueError("tool plan requires a string and nonnegative integer tool cap")
    value = json.loads(
        text,
        object_pairs_hook=_unique_object,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")),
    )
    canonical_json(value)
    if not isinstance(value, dict) or set(value) != {"actions", "response"}:
        raise ValueError("tool plan requires actions and response")
    if (
        not isinstance(value["actions"], list)
        or len(value["actions"]) > tool_cap
        or not isinstance(value["response"], str)
        or not value["response"].strip()
    ):
        raise ValueError("tool-plan actions/response violate the declared contract")
    for action in value["actions"]:
        if (
            not isinstance(action, dict)
            or set(action) != {"name", "arguments"}
            or not isinstance(action["name"], str)
            or action["name"] not in tool_names
            or not isinstance(action["arguments"], dict)
        ):
            raise ValueError("tool plan must use advertised tools and argument objects")
    return value


def verify_native_checkout(checkout, *, revision):
    root = Path(checkout).expanduser().resolve()
    if not root.is_dir():
        raise ExternalDependencyError("native benchmark checkout is unavailable")
    try:
        head = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise ExternalDependencyError("native benchmark must be a verified Git checkout") from exc
    if head != revision or dirty:
        raise ExternalDependencyError("native benchmark checkout revision or clean state differs")
    return root


@contextlib.contextmanager
def native_process_scope(root, *, package):
    """Native runtimes use process-global imports/CWD; serialize and restore them."""
    root = Path(root).resolve()
    with _NATIVE_LOCK:
        for name, module in tuple(sys.modules.items()):
            if name == package or name.startswith(package + "."):
                path = getattr(module, "__file__", None)
                if path and not Path(path).resolve().is_relative_to(root):
                    raise ExternalDependencyError(
                        "a different native benchmark checkout is imported"
                    )
        previous_cwd, previous_path = Path.cwd(), list(sys.path)
        try:
            os.chdir(root)
            sys.path[:0] = [str(root / "src"), str(root)]
            yield
        finally:
            sys.path[:] = previous_path
            os.chdir(previous_cwd)


def source_file_pins(root, paths):
    root = Path(root).resolve()
    result = {}
    for path in paths:
        path = Path(path).resolve()
        if not path.is_file() or not path.is_relative_to(root):
            raise ValidationError("native feedback source file must be inside its checkout")
        result[path.relative_to(root).as_posix()] = _sha(path)
    return result


def verify_source_files(root, files):
    if not isinstance(files, Mapping) or not files:
        raise ValidationError("native feedback needs pinned source files")
    for name, digest in files.items():
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or str(path) != name:
            raise ValidationError("native feedback source path must be relative")
        actual = (Path(root) / name).resolve()
        if not actual.is_relative_to(root) or not actual.is_file() or _sha(actual) != digest:
            raise ValidationError("native feedback source checksum changed")


class NativePlanFeedback:
    """Shared fit/stream validation and audited feedback, not a native agent."""

    def _initialize(self, *, source_input, checkout, output_root, mode):
        if mode not in ("offline", "online"):
            raise ValidationError("native ACE feedback mode must be offline or online")
        if not isinstance(source_input, Mapping) or set(source_input) != {"path", "sha256"}:
            raise ValidationError("native ACE feedback source needs a path and trusted SHA-256")
        self.source_input = dict(source_input)
        self.source = load_pinned_json_mapping(
            source_input["path"], expected_sha256=source_input["sha256"], max_bytes=16 * 1024 * 1024
        )
        self.mode, self.checkout = mode, Path(checkout).expanduser().resolve()
        rows = self.source.get("fitting_rows")
        required = {"case_id", "role", "state", "state_hash", "protected_checkpoint"}
        if (
            not isinstance(rows, list)
            or not rows
            or any(
                not isinstance(r, dict)
                or not required <= r.keys()
                or not isinstance(r["case_id"], str)
                or not isinstance(r["role"], str)
                for r in rows
            )
        ):
            raise ValidationError("native ACE feedback needs nonempty frozen fitting groups")
        self.rows = {r["case_id"]: r for r in rows}
        if len(self.rows) != len(rows) or any(
            not isinstance(k, str) or not k.strip() for k in self.rows
        ):
            raise ValidationError("native ACE fitting IDs must be unique strings")
        self.states = {key: _state(row["state"]) for key, row in self.rows.items()}
        if (
            set(row["role"] for row in rows) != {"train", "validation"}
            or any(self.states[k].state_hash != r["state_hash"] for k, r in self.rows.items())
            or len({s.state_hash for s in self.states.values()}) != len(rows)
            or set(self.rows) & set(self.source.get("heldout_ids", ()))
        ):
            raise ValidationError("native ACE fitting groups, states or heldout isolation differ")
        self.root = Path(output_root).expanduser().absolute()
        if self.root.resolve().is_relative_to(self.checkout):
            raise ValidationError("native ACE feedback output must be outside the clean checkout")
        if self.root.exists() or self.root.is_symlink():
            raise ValidationError("native ACE feedback output must be create-only")
        self.events, self.cache = [], {}

    def _create_output(self):
        validate_split_isolation(self.fit_examples("train"), self.fit_examples("validation"))
        self.root.mkdir(parents=True, exist_ok=False, mode=0o700)

    def _verify_source(self):
        original = load_pinned_json_mapping(
            self.source_input["path"],
            expected_sha256=self.source_input["sha256"],
            max_bytes=16 * 1024 * 1024,
        )
        if (
            original != self.source
            or self.rows != {r["case_id"]: r for r in original["fitting_rows"]}
            or set(self.states) != set(self.rows)
            or any(self.states[k].state_hash != r["state_hash"] for k, r in self.rows.items())
        ):
            raise ValidationError("native feedback in-memory source or states changed")
        verify_native_checkout(self.checkout, revision=self.source["native_source_revision"])
        verify_source_files(self.checkout, self.source["source_files"])

    def fit_examples(self, split):
        if split not in ("train", "validation"):
            raise ValidationError("native ACE fitting exposes no heldout examples")
        return tuple(
            self._example(key, split) for key, row in self.rows.items() if row["role"] == split
        )

    def validate_examples(self, train, validation):
        self.fingerprint()
        for supplied, split in ((train, "train"), (validation, "validation")):
            if tuple(to_jsonable(e) for e in supplied) != tuple(
                e.as_dict() for e in self.fit_examples(split)
            ):
                raise ValidationError("native ACE examples differ from the frozen fitting groups")

    def stream_examples(self):
        if self.mode != "online":
            raise ValidationError("native ACE stream feedback needs explicit online mode")
        return tuple(
            ACEExample.from_value(
                dict(
                    e.as_dict(),
                    metadata=dict(
                        e.as_dict()["metadata"],
                        original_operator_role=e.split,
                        source_official_split=self.source["source_official_split"],
                        online_protocol_label_NOT_official_test_split=True,
                    ),
                ),
                split="test",
            )
            for split in ("train", "validation")
            for e in self.fit_examples(split)
        )

    def validate_online_examples(self, supplied):
        self.fingerprint()
        if tuple(ACEExample.from_value(e, split="test").as_dict() for e in supplied) != tuple(
            e.as_dict() for e in self.stream_examples()
        ):
            raise ValidationError("native ACE online order, source or feedback targets changed")

    def score_answer(self, predicted, target):
        if not isinstance(target, str) or target not in self.rows:
            raise ValidationError("native ACE feedback target is outside the frozen fitting groups")
        fingerprint = self.fingerprint()
        state = self.states[target]
        try:
            plan = parse_native_tool_plan(
                predicted, tool_names=self.tool_names[target], tool_cap=state.budget.max_tool_calls
            )
        except (ValueError, TypeError):
            self.events.append(
                dict(
                    target=target,
                    prediction_sha256=content_hash(predicted),
                    status="invalid_tool_plan_training_feedback",
                    score=0.0,
                    source_state_hash=state.state_hash,
                    native_task_score_NOT_claimed=True,
                    native_metrics=None,
                    new_environment_calls=0,
                )
            )
            return 0.0
        key = content_hash([target, state.state_hash, plan, fingerprint])
        with _NATIVE_LOCK:
            reused = key in self.cache
            if not reused:
                folder = self.root / "native-feedback" / key
                folder.mkdir(parents=True, exist_ok=False)
                self.cache[key] = self._native_score(target, plan, folder)
            result = self.cache[key]
        score = result["native_metrics"][self.metric]
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValidationError("native ACE task feedback score is invalid")
        self.events.append(
            dict(
                target=target,
                prediction_sha256=content_hash(predicted),
                score=score,
                result=result,
                source_state_hash=state.state_hash,
                exact_feedback_reuse_NOT_new_model_draw=reused,
            )
        )
        return float(score)

    def answer_is_correct(self, predicted, ground_truth):
        return self.score_answer(predicted, ground_truth) > 0

    def evaluate_accuracy(self, predictions, ground_truths):
        if len(predictions) != len(ground_truths):
            raise ValidationError("native ACE prediction/target lengths differ")
        return (
            sum(self.score_answer(p, t) for p, t in zip(predictions, ground_truths, strict=True))
            / len(predictions)
            if predictions
            else 0.0
        )
