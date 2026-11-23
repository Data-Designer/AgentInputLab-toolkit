from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agentinputlab.contracts import BudgetExceededError, ExternalDependencyError, ValidationError
from agentinputlab.recipes import (
    SmartSearchProcessTeacher,
    SmartSearchTeacherCompletion,
    SmartSearchTeacherControls,
    SmartSearchTrainingCase,
)


class TeacherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.source_binding = {"root": str(self.source.resolve()), "files": {}, "fixture_only": True}
        self.train = [SmartSearchTrainingCase("t", "Train question", "train")]
        self.dev = [SmartSearchTrainingCase("d", "Dev question", "validation")]
        self.rows = [{"question": "Train question", "golden_answers": ["Gold"],
                      "output": {"final_response": "recorded fixture response", "pred": "Gold",
                                 "metric_score": {"f1": 1.0}}}]
        self.calls = []

    def tearDown(self):
        self.temp.cleanup()

    def provider(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return SmartSearchTeacherCompletion("recorded fixture teacher output", 20, 10, "a" * 64)

    def native(self, root, key, source):
        if key == "reward":
            instance = SimpleNamespace()

            def evaluate(rows):
                answer = instance.generate_response("native fixture reward prompt")
                return [{"question": rows[0]["question"], "golden_answer": "Gold",
                         "response": rows[0]["output"]["final_response"],
                         "process_scores": [{"score": "0", "original_content": answer}],
                         "outcome_score": rows[0]["output"]["metric_score"]}]

            instance.evaluate_search = evaluate
            return instance
        if key == "redundancy":
            return SimpleNamespace(detect_redundancy=lambda rows: rows)
        if key == "refinement":
            instance = SimpleNamespace()

            def rewrite(rows):
                rows[0]["process_scores"][0]["refined"] = instance.generate_response("native fixture refine prompt")
                return rows

            instance.rewrite_search = rewrite
            return instance
        raise AssertionError(key)

    def run_teacher(self, *, provider=None, controls=None, **kwargs):
        teacher = SmartSearchProcessTeacher(provider or self.provider,
                                           model_fingerprint_sha256="a" * 64, controls=controls)
        arguments = dict(checkout=self.source, output_root=self.root / "out", train=self.train,
                         validation=self.dev, sampled_rollouts=self.rows,
                         rollout_provenance={"fixture_only": True, "not_real_acceptance": True})
        arguments.update(kwargs)
        with patch("agentinputlab.recipes.smartsearch_teacher.verify_smartsearch_preparation_source",
                   return_value=self.source_binding), patch(
                       "agentinputlab.recipes.smartsearch_teacher._native_converter", side_effect=self.native):
            return teacher.annotate(**arguments)

    def test_retains_native_prompts_controls_outputs_and_measured_costs(self):
        before = json.dumps(self.rows)
        result = self.run_teacher()
        self.assertEqual(json.dumps(self.rows), before)
        self.assertEqual(result["usage"]["model_attempts"], 2)
        self.assertEqual(result["usage"]["model_responses"], 2)
        self.assertEqual(result["usage"]["output_tokens"], 20)
        self.assertEqual(result["usage"]["unknown_output_token_reservations"], 0)
        self.assertFalse(result["neural_training_performed"])
        self.assertEqual(self.calls[0], ("native fixture reward prompt", {
            "temperature": 0.6, "max_new_tokens": 2048, "stop": (), "seed": 17,
        }))
        self.assertEqual(self.calls[1][1]["stop"], ("</search>",))
        rows = json.loads((self.root / "out/scored-rollouts.json").read_text())
        self.assertEqual(rows[0]["process_scores"][0]["refined"],
                         "recorded fixture teacher output</search>")
        with self.assertRaises(ValidationError):
            self.run_teacher()
        self.assertEqual(len(self.calls), 2)

    def test_native_cap_is_not_shrunk_to_fit_budget(self):
        with self.assertRaises(BudgetExceededError):
            self.run_teacher(controls=SmartSearchTeacherControls(max_total_output_tokens=1024))
        self.assertEqual(self.calls, [])
        failed = json.loads((self.root / "out/failed.json").read_text())
        self.assertEqual(failed["usage"]["model_attempts"], 0)

    def test_provider_failure_is_not_empty_score_or_retry(self):
        def failing(prompt, **kwargs):
            self.calls.append(prompt)
            raise ConnectionError("fixture provider failure")

        with self.assertRaises(ConnectionError):
            self.run_teacher(provider=failing)
        self.assertEqual(len(self.calls), 1)
        failed = json.loads((self.root / "out/failed.json").read_text())
        self.assertEqual(failed["usage"]["model_attempts"], 1)
        self.assertEqual(failed["usage"]["model_responses"], 0)
        self.assertEqual(failed["usage"]["unknown_output_token_reservations"], 2048)
        self.assertFalse((self.root / "out/scored-rollouts.json").exists())

    def test_real_response_usage_is_retained_before_identity_or_cap_failure(self):
        for fingerprint, output, error in (("b" * 64, 10, ExternalDependencyError),
                                           ("a" * 64, 2049, BudgetExceededError)):
            with self.subTest(output=output):
                def response(prompt, output=output, fingerprint=fingerprint, **kwargs):
                    return SmartSearchTeacherCompletion("fixture", 20, output, fingerprint)

                out = self.root / f"out-{output}"
                with self.assertRaises(error):
                    self.run_teacher(provider=response, output_root=out)
                failed = json.loads((out / "failed.json").read_text())
                self.assertEqual(failed["usage"]["output_tokens"], output)
                self.assertTrue((out / "call-0001-response.json").is_file())

    def test_no_heldout_rewards_dispatched(self):
        for bad in ({**self.rows[0], "question": "Dev question"},
                    {**self.rows[0], "golden_answers": []},
                    {**self.rows[0], "output": {**self.rows[0]["output"], "metric_score": {"f1": True}}}):
            with self.subTest(bad=bad), self.assertRaises(ValidationError):
                self.run_teacher(sampled_rollouts=[bad])
        self.assertEqual(self.calls, [])
        self.assertFalse((self.root / "out").exists())

    def test_source_output_overlap_rejected_before_provider(self):
        with self.assertRaises(ValidationError):
            self.run_teacher(output_root=self.source / "outputs")
        self.assertEqual(self.calls, [])

    def test_invalid_controls_and_unmetered_response_rejected(self):
        for values in ({"seed": True}, {"max_model_calls": 0}, {"max_total_output_tokens": False}):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                SmartSearchTeacherControls(**values)
        with self.assertRaises(ExternalDependencyError):
            self.run_teacher(provider=lambda *args, **kwargs: "unmetered fixture")
        failed = json.loads((self.root / "out/failed.json").read_text())
        self.assertEqual(failed["usage"]["unknown_output_token_reservations"], 2048)


if __name__ == "__main__":
    unittest.main()
