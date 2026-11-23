from __future__ import annotations

import sys
import unittest

from agentinputlab.contracts import ExternalDependencyError, WorkerRequest, WorkerResponse
from agentinputlab.runner import FunctionWorkerClient, JsonLineWorkerClient


class WorkerProtocolTests(unittest.TestCase):
    def test_function_worker_keeps_request_identity(self) -> None:
        client = FunctionWorkerClient(
            lambda request: WorkerResponse(
                schema_version=1,
                request_id=request.request_id,
                status="ok",
                payload={"seen": request.release_id},
            )
        )
        response = client.invoke(
            WorkerRequest(
                schema_version=1,
                release_id="r",
                request_id="id",
                payload={},
            )
        )
        self.assertEqual(response.payload["seen"], "r")

    def test_subprocess_worker_uses_single_json_line(self) -> None:
        script = (
            "import json,sys;"
            "r=json.loads(sys.stdin.readline());"
            "print(json.dumps({'schema_version':1,'request_id':r['request_id'],"
            "'status':'ok','payload':{'value':1},'error_code':'','error_message':''}))"
        )
        client = JsonLineWorkerClient((sys.executable, "-c", script))
        response = client.invoke(
            WorkerRequest(
                schema_version=1,
                release_id="r",
                request_id="id",
                payload={},
            )
        )
        self.assertEqual(response.payload["value"], 1)

    def test_subprocess_worker_rejects_mismatched_request_identity(self) -> None:
        script = (
            "import json,sys;"
            "json.loads(sys.stdin.readline());"
            "print(json.dumps({'schema_version':1,'request_id':'wrong',"
            "'status':'ok','payload':{},'error_code':'','error_message':''}))"
        )
        client = JsonLineWorkerClient((sys.executable, "-c", script))
        with self.assertRaisesRegex(ExternalDependencyError, "request_id mismatch"):
            client.invoke(
                WorkerRequest(
                    schema_version=1,
                    release_id="r",
                    request_id="expected",
                    payload={},
                )
            )


if __name__ == "__main__":
    unittest.main()
