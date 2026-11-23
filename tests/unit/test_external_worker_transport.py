from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

from agentinputlab.contracts import ExternalDependencyError
from agentinputlab.integrations.external_workers.base import JsonLineSubprocessWorker
from agentinputlab.integrations.external_workers.supervisor import SupervisedWorkerError


class JsonLineWorkerTests(unittest.TestCase):
    def test_one_object_round_trip_without_shell(self):
        code = (
            "import json,sys; value=json.loads(sys.stdin.readline()); "
            "print(json.dumps({'ok': value['value']}))"
        )
        worker = JsonLineSubprocessWorker(
            command=(sys.executable, "-c", code),
            worker_id="echo",
            identity={"revision": "fixture"},
        )
        self.assertEqual(worker.call({"value": 7}), {"ok": 7})
        self.assertIsNotNone(worker.last_resource_receipt)
        self.assertEqual(
            worker.last_resource_receipt.limit_hash,
            worker.resource_limits.limit_hash,
        )
        self.assertEqual(worker.fingerprint()["protocol"], "agentinputlab.jsonl.v1")

    def test_multiple_stdout_records_are_rejected(self):
        worker = JsonLineSubprocessWorker(
            command=(sys.executable, "-c", "print('{}'); print('{}')"),
            worker_id="bad",
            identity={"revision": "fixture-v1"},
        )
        with self.assertRaisesRegex(SupervisedWorkerError, "exactly one JSON line") as raised:
            worker.call({})
        self.assertIs(raised.exception.resource_receipt, worker.last_resource_receipt)

    def test_nonzero_exit_is_typed(self):
        worker = JsonLineSubprocessWorker(
            command=(sys.executable, "-c", "raise SystemExit(3)"),
            worker_id="failed",
            identity={"revision": "fixture-v1"},
        )
        with self.assertRaisesRegex(SupervisedWorkerError, "code 3") as raised:
            worker.call({})
        self.assertEqual(raised.exception.resource_receipt.exit_code, 3)
        self.assertIs(raised.exception.resource_receipt, worker.last_resource_receipt)

    def test_stdout_is_stopped_at_the_streaming_limit(self):
        worker = JsonLineSubprocessWorker(
            command=(sys.executable, "-c", "import sys; sys.stdout.write('x' * 1000000)"),
            worker_id="flood",
            identity={"revision": "fixture-v1"},
            max_output_bytes=128,
        )
        with self.assertRaisesRegex(ExternalDependencyError, "stdout limit"):
            worker.call({})

    def test_parent_secrets_are_not_inherited_by_default(self):
        code = (
            "import json,os; "
            "print(json.dumps({'leaked': os.getenv('AGENTINPUTLAB_TEST_SECRET') is not None}))"
        )
        worker = JsonLineSubprocessWorker(
            command=(sys.executable, "-c", code),
            worker_id="environment",
            identity={"revision": "fixture-v1"},
        )
        with patch.dict(os.environ, {"AGENTINPUTLAB_TEST_SECRET": "do-not-inherit"}):
            self.assertEqual(worker.call({}), {"leaked": False})

    def test_explicit_worker_environment_is_available(self):
        code = "import json,os; print(json.dumps({'value': os.getenv('EXPLICIT_FIXTURE')}))"
        worker = JsonLineSubprocessWorker(
            command=(sys.executable, "-c", code),
            worker_id="environment",
            identity={"revision": "fixture-v1"},
            environment={"EXPLICIT_FIXTURE": "visible"},
        )
        self.assertEqual(worker.call({}), {"value": "visible"})

    def test_passed_environment_is_explicit_and_secret_free(self):
        code = "import json,os; print(json.dumps({'value': os.getenv('PASSED_FIXTURE')}))"
        worker = JsonLineSubprocessWorker(
            command=(sys.executable, "-c", code),
            worker_id="passed-environment",
            identity={"revision": "fixture-v1"},
            pass_environment=("PASSED_FIXTURE",),
        )
        with patch.dict(os.environ, {"PASSED_FIXTURE": "credential-value"}):
            self.assertEqual(worker.call({}), {"value": "credential-value"})
            encoded = str(worker.fingerprint())
        self.assertIn("PASSED_FIXTURE", encoded)
        self.assertNotIn("credential-value", encoded)

    def test_missing_passed_environment_fails_before_spawn(self):
        worker = JsonLineSubprocessWorker(
            command=(sys.executable, "-c", "print('{}')"),
            worker_id="missing-environment",
            identity={"revision": "fixture-v1"},
            pass_environment=("MISSING_PASSED_FIXTURE",),
        )
        with (
            patch.dict(os.environ, {}, clear=True),
            self.assertRaisesRegex(ExternalDependencyError, "MISSING_PASSED_FIXTURE"),
        ):
            worker.call({})

    def test_worker_fingerprint_covers_execution_identity(self):
        common = {
            "worker_id": "fixture",
            "identity": {
                "revision": "v1",
                "protocol": "cannot-override",
                "worker_id": "cannot-override",
            },
        }
        first = JsonLineSubprocessWorker(
            command=(sys.executable, "-c", "print('{}')"),
            environment={"ROUTE": "a"},
            **common,
        )
        second = JsonLineSubprocessWorker(
            command=(sys.executable, "-c", "print('{\"other\": 1}')"),
            environment={"ROUTE": "b"},
            **common,
        )
        self.assertNotEqual(first.fingerprint(), second.fingerprint())
        self.assertEqual(first.fingerprint()["worker_id"], "fixture")
        self.assertEqual(first.fingerprint()["protocol"], "agentinputlab.jsonl.v1")


if __name__ == "__main__":
    unittest.main()
