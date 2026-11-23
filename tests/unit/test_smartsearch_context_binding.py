import pytest

from agentinputlab import ExecutionBudget, FrozenAgentState, InterventionContext
from agentinputlab.contracts import ExternalDependencyError
from agentinputlab.recipes import SmartSearchArtifactIdentity, SmartSearchCommonSupportQueries
from agentinputlab.runner import InterventionRunner


def test_smartsearch_worker_is_injected_through_context_not_hidden_on_intervention():
    artifact = SmartSearchArtifactIdentity(
        model_snapshot_manifest_sha256="a" * 64,
        model_snapshot_tree_sha256="b" * 64,
        tokenizer_revision="82bfe8a845e3e08f0d9d00d8e1e2cfb6d50b5da8",
        served_model_name="fixture",
        model_deployment_fingerprint_sha256="c" * 64,
        context_admission_fingerprint_sha256="d" * 64,
        retriever_revision="fixture",
        retriever_deployment_fingerprint_sha256="e" * 64,
        corpus_sha256="f" * 64,
        index_sha256="1" * 64,
        retrieval_config_sha256="2" * 64,
        compatibility_manifest_sha256="3" * 64,
    )

    class Worker:
        worker_id = "non-generating-fixture"
        calls = 0

        def fingerprint(self):
            return {"worker_id": self.worker_id}

        def call(self, payload):
            self.calls += 1
            return {
                "status": "error",
                "failure_code": "FIXTURE_BOUNDARY",
                "error_message": "non-generating fixture worker reached",
                "consumed": False,
            }

    client = Worker()
    with pytest.raises(ExternalDependencyError, match="FIXTURE_BOUNDARY"):
        InterventionRunner().run(
            SmartSearchCommonSupportQueries(artifact),
            FrozenAgentState(query="q", budget=ExecutionBudget(max_output_tokens=8192)),
            InterventionContext(external_runtime=client, seed=17),
        )
    assert client.calls == 1
