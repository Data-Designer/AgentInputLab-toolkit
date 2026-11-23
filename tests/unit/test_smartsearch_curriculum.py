from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agentinputlab.contracts import ExternalDependencyError, ValidationError
from agentinputlab.recipes.smartsearch_curriculum import (
    SmartSearchCurriculumPreparation,
    SmartSearchTrainingCase,
    validate_smartsearch_training_splits,
    verify_smartsearch_preparation_source,
)


class CurriculumTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        programs = {
            "detect_redundancy.py": "class Redundancy:\n def detect_redundancy(self,rows):\n  for r in rows:r['format']=int('valid' in r['response']);r['redundancy']=[]\n  return rows\n",
            "transfer_sft.py": "class Transfer:\n def transfer(self,rows):\n  return [dict(instruction='fixture',input=r['question'],output=r['response']) for r in rows if r['format'] and r['outcome_score']['f1']>=0.8]\n",
            "transfer_dpo.py": "class Transfer:\n def transfer(self,before,after):\n  return [dict(input=a['question'],chosen=a['response'],rejected=b['response']) for a in before for b in after if a['question']==b['question'] and a['outcome_score']['f1']>b['outcome_score']['f1']]\n",
            "transfer_regenerate.py": "class Transfer:\n def transfer(self,rows):\n  return []\n",
        }
        files = {}
        for name, body in programs.items():
            relative = "scripts/data_construction/" + name
            path = self.source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body)
            files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.fingerprint = {"source_commit": "fixture", "files": files,
                            "root": str(self.source.resolve()), "preparation_math_modified": False}
        self.train = (SmartSearchTrainingCase("train-1", "Training question", "train"),)
        self.validation = (SmartSearchTrainingCase("dev-1", "Development question", "validation"),)
        self.initial = [{"question": "Training question", "response": "valid recorded response",
                         "process_scores": [], "outcome_score": {"f1": 1.0}}]
        self.after = [{"question": "Training question", "response": "recorded failed response",
                       "process_scores": [], "outcome_score": {"f1": 0.0}}]

    def tearDown(self):
        self.temp.cleanup()

    def prepare(self, **kwargs):
        arguments = dict(checkout=self.source, output_root=self.root / "out", train=self.train,
                         validation=self.validation, initial_scored_rollouts=self.initial,
                         regenerated_scored_rollouts=self.after,
                         rollout_provenance={"fixture_only": True, "model_calls_in_fixture": 0})
        arguments.update(kwargs)
        with patch("agentinputlab.recipes.smartsearch_curriculum.verify_smartsearch_preparation_source",
                   return_value=self.fingerprint):
            return SmartSearchCurriculumPreparation().prepare(**arguments)

    def test_uses_native_converter_outputs_without_fitting_or_mutating_inputs(self):
        before = json.dumps(self.initial, sort_keys=True)
        result = self.prepare()
        self.assertEqual((result.sft_count, result.dpo_count, result.prefix_count), (1, 1, 0))
        self.assertEqual(json.dumps(self.initial, sort_keys=True), before)
        manifest = json.loads((Path(result.root) / "manifest.json").read_text())
        self.assertFalse(manifest["neural_training_performed"])
        self.assertEqual(manifest["model_calls_performed"], 0)
        self.assertTrue(manifest["empty_selection_is_not_training_success"])
        self.assertFalse(tuple(self.source.rglob("*.pyc")))
        for name, expected in result.output_hashes.items():
            self.assertEqual(hashlib.sha256((Path(result.root) / name).read_bytes()).hexdigest(), expected)
        with self.assertRaises(ValidationError):
            self.prepare()

    def test_rejects_cross_split_ids_and_normalized_questions(self):
        for dev in ((SmartSearchTrainingCase("train-1", "Other", "validation"),),
                    (SmartSearchTrainingCase("different", "  training  QUESTION ", "validation"),)):
            with self.subTest(dev=dev), self.assertRaises(ValidationError):
                validate_smartsearch_training_splits(self.train, dev)

    def test_forbids_heldout_reward_rows_before_dispatch(self):
        bad = [{**self.initial[0], "question": "Development question"}]
        with self.assertRaises(ValidationError):
            self.prepare(initial_scored_rollouts=bad)
        self.assertFalse((self.root / "out").exists())

    def test_empty_native_selection_does_not_become_training_success(self):
        result = self.prepare(initial_scored_rollouts=self.after, regenerated_scored_rollouts=[])
        self.assertEqual((result.sft_count, result.dpo_count), (0, 0))
        self.assertFalse(json.loads((Path(result.root) / "manifest.json").read_text())["neural_training_performed"])

    def test_forbids_source_output_overlap_including_symlinked_parent(self):
        link = self.root / "source-link"
        link.symlink_to(self.source, target_is_directory=True)
        for out in (self.source / "new-output", link / "new-output"):
            with self.subTest(out=out), self.assertRaises(ValidationError):
                self.prepare(output_root=out)
        self.assertFalse((self.source / "new-output").exists())

    def test_checks_source_bytes_before_compiling(self):
        (self.source / "scripts/data_construction/transfer_sft.py").write_text("raise RuntimeError('unbound')")
        with self.assertRaises(ExternalDependencyError):
            self.prepare()
        failure = json.loads((self.root / "out/failed.json").read_text())
        self.assertEqual(failure["model_calls_performed"], 0)

    def test_rejects_invalid_measured_scores_and_caps(self):
        for f1 in (None, True, float("nan"), -1, 2):
            with self.subTest(f1=f1), self.assertRaises(ValidationError):
                self.prepare(initial_scored_rollouts=[{**self.initial[0], "outcome_score": {"f1": f1}}])
        with self.assertRaises(ValidationError):
            SmartSearchCurriculumPreparation(max_rollout_rows=True)
        with self.assertRaises(ExternalDependencyError):
            verify_smartsearch_preparation_source(self.source)


if __name__ == "__main__":
    unittest.main()
