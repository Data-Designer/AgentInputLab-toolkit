from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from agentinputlab.contracts import ExternalDependencyError, ValidationError
from agentinputlab.recipes import (
    SKILLOPT_PROTOCOL,
    SKILLOPT_RELEASE_ID,
    SKILLOPT_UPSTREAM_COMMIT,
    SkillOptControls,
    SkillOptOfflineRecipe,
    load_skillopt_artifact,
    validate_skillopt_checkout,
)


class SkillOptRecipeTests(unittest.TestCase):
    def test_checkout_validator_rejects_missing_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing"
            with self.assertRaisesRegex(ExternalDependencyError, "regular directory"):
                validate_skillopt_checkout(missing)

    def test_command_preserves_upstream_entrypoint_and_fairness_controls(self) -> None:
        controls = SkillOptControls(
            num_epochs=3,
            batch_size=12,
            learning_rate=2,
            seed=7,
            max_steps=5,
        )
        command = SkillOptOfflineRecipe(
            controls=controls,
            python_executable="python-fixture",
        ).build_command(
            checkout="/tmp/skillopt-source",
            config_path="/tmp/skillopt-config.yaml",
            output_root="/tmp/skillopt-output",
        )
        self.assertEqual(command[0], "python-fixture")
        self.assertTrue(command[1].endswith("/scripts/train.py"))
        self.assertEqual(command.count("--cfg-options"), 1)
        self.assertIn("train.num_epochs=3", command)
        self.assertIn("train.batch_size=12", command)
        self.assertIn("optimizer.learning_rate=2", command)
        self.assertIn("evaluation.use_gate=true", command)
        self.assertIn("env.max_steps=5", command)

    def test_completed_output_becomes_content_addressed_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payloads = {
                "best_skill.md": "# Skill\n\nVerify evidence before acting.\n",
                "config.json": json.dumps({"env": "fixture"}),
                "history.json": json.dumps([{"step": 1, "score": 0.8}]),
                "runtime_state.json": json.dumps({"best_step": 1}),
            }
            for relative, text in payloads.items():
                (root / relative).write_text(text, encoding="utf-8")
            artifact = load_skillopt_artifact(
                root,
                config_sha256="a" * 64,
                controls={"seed": 11, "learning_rate": 4},
                model_fingerprints={
                    "target": {"model": "target-fixture"},
                    "optimizer": {"model": "optimizer-fixture"},
                },
                environment_keys=("OPENAI_API_KEY",),
                command_sha256="b" * 64,
            )
        self.assertEqual(artifact.release_id, SKILLOPT_RELEASE_ID)
        self.assertEqual(artifact.upstream_commit, SKILLOPT_UPSTREAM_COMMIT)
        self.assertEqual(artifact.protocol, SKILLOPT_PROTOCOL)
        self.assertEqual(
            artifact.best_skill_sha256,
            hashlib.sha256(payloads["best_skill.md"].encode()).hexdigest(),
        )
        self.assertEqual(len(artifact.artifact_id), 64)
        self.assertEqual(artifact.environment_keys, ("OPENAI_API_KEY",))
        self.assertNotIn("secret", artifact.as_dict())

    def test_missing_validation_gate_output_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "best_skill.md").write_text("skill", encoding="utf-8")
            with self.assertRaisesRegex(Exception, "config.json"):
                load_skillopt_artifact(
                    root,
                    config_sha256="a" * 64,
                    controls={},
                    model_fingerprints={"target": {}, "optimizer": {}},
                    command_sha256="b" * 64,
                )

    def test_invalid_controls_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValidationError, "seed"):
            SkillOptControls(seed=-1)
        with self.assertRaisesRegex(ValidationError, "max_steps"):
            SkillOptControls(max_steps=0)

    def test_timeout_rejects_invalid_or_unrepresentable_numbers(self) -> None:
        invalid = (
            True, False, 0, -1, float("nan"), float("inf"), float("-inf"),
            "900", None, 10**10000,
        )
        for index, value in enumerate(invalid):
            with self.subTest(index=index), self.assertRaisesRegex(
                ValidationError, "finite and positive"
            ):
                SkillOptControls(timeout_seconds=value)

    def test_positive_finite_timeout_preserves_configured_value(self) -> None:
        for value in (0.1, 1, 900, 7200.0):
            with self.subTest(timeout=value):
                controls = SkillOptControls(timeout_seconds=value)
                self.assertEqual(controls.timeout_seconds, value)
                self.assertIs(type(controls.timeout_seconds), type(value))


if __name__ == "__main__":
    unittest.main()
