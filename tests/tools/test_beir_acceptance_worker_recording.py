import unittest
from unittest.mock import patch

from run_beir_archive_acceptance import RecordingWorker, preflight_dependencies

from agentinputlab.contracts import ExternalDependencyError


class RecordingWorkerTests(unittest.TestCase):
    def test_missing_declared_extra_is_rejected_before_dispatch(self):
        with patch("run_beir_archive_acceptance.importlib.util.find_spec", return_value=None):
            for release, extra in [("rewrite.query2doc.querygym_live", "querygym"),
                                   ("rewrite.reformer.repo_native_r_only", "reformer")]:
                with self.assertRaisesRegex(ExternalDependencyError, extra):
                    preflight_dependencies((release,))
            preflight_dependencies(("baseline.no_op",))

    def test_declared_failure_is_recorded_unchanged_including_usage(self):
        class Worker:
            worker_id = "native-fixture"
            last_resource_receipt = None

            def fingerprint(self):
                return {"source": "fixture"}

            def call(self, payload):
                self.payload = payload
                return {"status": "failed", "failure_code": "TEXT_INVALID",
                        "usage": {"model_calls": 1, "input_tokens": 10, "output_tokens": 512}}

        inner, exchanges, payload = Worker(), [], {"request_id": "fixed"}
        worker = RecordingWorker(inner, exchanges)
        response = worker.call(payload)
        self.assertIs(inner.payload, payload)
        self.assertEqual(exchanges[0]["response"], response)
        self.assertEqual(exchanges[0]["response"]["usage"]["output_tokens"], 512)
        self.assertEqual(worker.fingerprint(), inner.fingerprint())
        self.assertIsNone(worker.last_resource_receipt)

    def test_transport_error_is_not_retried(self):
        class Worker:
            worker_id = "native-fixture"
            last_resource_receipt = None
            calls = 0

            def call(self, payload):
                self.calls += 1
                raise RuntimeError("fixture transport failure")

        inner, exchanges = Worker(), []
        with self.assertRaisesRegex(RuntimeError, "transport failure"):
            RecordingWorker(inner, exchanges).call({"request_id": "fixed"})
        self.assertEqual(inner.calls, 1)
        self.assertEqual(exchanges[0]["error_type"], "RuntimeError")
        self.assertNotIn("response", exchanges[0])
