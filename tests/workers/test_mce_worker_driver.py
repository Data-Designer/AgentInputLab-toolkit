from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import workers.mce.driver as mce_driver
from agentinputlab.contracts import (
    AlgorithmOutputError,
    ExternalDependencyError,
    InterventionContext,
)
from agentinputlab.contracts.serialization import content_hash, to_jsonable
from agentinputlab.recipes.mce import (
    MCE_EMPTY_INITIAL_CONTEXT_SHA256,
    MCEContextArtifact,
    MCEControls,
    MCEExample,
    MCEOfflineRecipe,
    validate_mce_split_isolation,
)
from agentinputlab.recipes.mce_final import (
    MCEFinalArtifactEvaluator,
    MCEFinalControls,
    MCEFinalExample,
    validate_final_summary,
    validate_output_budget,
)
from workers.mce.driver import (
    EMPTY_INITIAL_CONTEXT,
    PROVIDER_SUBCALL_SEED_MODE,
    SEED_MODE,
    SEMANTIC_PATCH_MANIFEST,
    SEMANTIC_PATCH_MANIFEST_SHA256,
    UPSTREAM_COMMIT,
    UPSTREAM_MAIN_SHA256,
    DriverError,
    MCEEvolutionIteration,
    MCEEvolutionRun,
    MCEOfficialRuntime,
    MCEWorkerDriver,
    PersistentArtifactStore,
    TrustedEvaluationSupervisor,
    TrustedProviderSupervisor,
    _content_hash,
    _interface_source_manifest,
    _validate_model_fingerprints,
    derive_provider_subcall_seed,
    final_evaluation_manager_contract,
    serve_jsonl,
)

_IMAGE = "sha256:" + "1" * 64
_INTERFACE = {
    "environment": "fixture-environment",
    "interfaces": (
        {
            "name": "retrieve_context",
            "signature": "retrieve_context(query: str) -> str",
        },
    ),
}
_ENVIRONMENT = {
    "name": "fixture-environment",
    "revision": "fixture-environment@v1",
    "source_sha256": "2" * 64,
}
_MODELS = {
    role: {
        "provider": "fixture-provider",
        "model": f"{role}-fixture",
        "revision": f"{role}@v1",
        "configuration_sha256": character * 64,
    }
    for role, character in (("target", "3"), ("meta", "4"), ("base", "5"))
}


def _sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture_checkout(root: Path) -> Path:
    files = {
        "mce/main.py": "async def run_iteration(**kwargs):\n    return kwargs\n",
        "mce/base_agent.py": "def run_base_agent():\n    return None\n",
        "mce/meta_agent.py": "def run_meta_agent():\n    return None\n",
        "mce/eval.py": "def batch_evaluate():\n    return None\n",
        "mce/validation.py": "def validate_interfaces():\n    return None\n",
        "mce/utils.py": "def setup_meta_agent_reference():\n    return None\n",
        "env/base.py": "class InterfaceSignature:\n    pass\n",
        "env/registry.py": "class EnvironmentRegistry:\n    pass\n",
    }
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    license_path = root / "LICENSE"
    license_path.write_text("MIT License\nfixture\n", encoding="utf-8")
    source_files = {relative: _sha256_file(root / relative) for relative in files}
    manifest = {
        "schema": "agentinputlab.mce.compatibility.v2",
        "source": {
            "repository": "fixture://meta-context-engineering",
            "commit": UPSTREAM_COMMIT,
            "tree": "6" * 40,
            "fixture": True,
            "required_files": source_files,
            "import_closure": source_files,
        },
        "license": {
            "schema": "agentinputlab.mce.license.v1",
            "license_id": "MIT",
            "path": "LICENSE",
            "sha256": _sha256_file(license_path),
        },
        "provenance": {
            "schema": "agentinputlab.mce.provenance.v1",
            "repository": "fixture://meta-context-engineering",
            "commit": UPSTREAM_COMMIT,
            "tree": "6" * 40,
            "implementation": "fixture_external_worker",
            "paper_result_parity": False,
        },
        "semantic_patch": SEMANTIC_PATCH_MANIFEST,
        "algorithm_contract": {
            "baseline_iteration": 0,
            "baseline_artifact": "canonical_empty",
            "evolved_skill_required": True,
            "selection": "max_primary_metric_first_tie",
        },
        "security": {
            "validation_oracle_mode": "evaluator_only",
            "usage_accounting_mode": "trusted_supervisor_ipc",
            "pre_call_budget_enforced": True,
            "artifact_store": "persistent_content_addressed_v1",
            "artifact_egress": "manifest_and_opaque_handle_only",
            "paper_result_parity_claimed": False,
        },
    }
    manifest_path = root / "compatibility_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return manifest_path


def _sandbox() -> dict[str, Any]:
    return {
        "attestation_id": "mce-fixture-attestation-v2",
        "isolation_kind": "container",
        "image_digest": _IMAGE,
        "policy_sha256": "7" * 64,
        "test_report_sha256": "8" * 64,
        "network_mode": "none",
        "dataset_mount_mode": "read_only",
        "workspace_scope": "per_run",
        "generated_code_execution": "isolated_worker_only",
        "test_status": "passed",
        "validation_oracle_mode": "evaluator_only",
        "oracle_mount_mode": "supervisor_only_hidden",
        "oracle_process_boundary": "trusted_supervisor_out_of_process",
        "usage_accounting_mode": "trusted_supervisor_ipc",
        "provider_guard_mode": "provider_transport_pre_call",
        "seed_mode": SEED_MODE,
        "provider_subcall_seed_mode": PROVIDER_SUBCALL_SEED_MODE,
        "host_secrets_mounted": False,
        "structured_egress_only": True,
        "generated_agents_validation_access": False,
        "generated_workspace_oracle_visible": False,
        "pre_call_budget_enforced": True,
        "wall_timeout_enforced": True,
        "source_read_only": True,
        "artifact_store_persistent": True,
        "model_identity_enforced": True,
        "environment_identity_verified": True,
        "interface_signature_verified": True,
        "semantic_patch_applied": True,
        "semantic_patch_manifest_sha256": SEMANTIC_PATCH_MANIFEST_SHA256,
        "upstream_run_iteration_source_sha256": UPSTREAM_MAIN_SHA256,
        "interface_validation_mode": "remote_static_no_import",
        "batch_evaluation_mode": "trusted_supervisor_no_generated_import",
        "resource_limits": {
            "cpu_seconds": 600,
            "memory_bytes": 2 * 1024 * 1024 * 1024,
            "processes": 64,
            "wall_seconds": 600,
        },
        "required_worker_contract": {
            "oracle_mount_mode": "supervisor_only_hidden",
            "oracle_process_boundary": "trusted_supervisor_out_of_process",
            "generated_workspace_oracle_visible": False,
            "provider_guard_mode": "provider_transport_pre_call",
            "seed_mode": SEED_MODE,
            "provider_subcall_seed_mode": PROVIDER_SUBCALL_SEED_MODE,
            "validation_scoring_mode": "supervisor_only",
        },
    }


class FixtureRuntime:
    def __init__(self, *, missing_skill: int | None = None, over_budget: bool = False) -> None:
        self.calls = 0
        self.provider_calls = 0
        self.missing_skill = missing_skill
        self.over_budget = over_budget
        self.workspace: Path | None = None
        self.validation_path: Path | None = None
        self.train_records: tuple[dict[str, Any], ...] = ()
        self.score_provider = lambda _index: 0.0

    def evolve(
        self,
        *,
        workspace,
        train_path,
        validation_path,
        controls,
        model_fingerprints,
        supervisor,
    ):
        self.calls += 1
        self.workspace = workspace
        self.validation_path = validation_path
        validation_path.relative_to(validation_path.parent)
        with self.assert_outside(workspace, validation_path):
            pass
        assert not tuple(workspace.rglob("*validation*"))
        assert model_fingerprints == _MODELS
        self.train_records = tuple(
            json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()
        )
        count = 2 if self.over_budget else controls["iterations"] + 1
        for index in range(count):
            supervisor.before_model_call(input_tokens=10 + index)
            self.provider_calls += 1
            supervisor.after_model_call(output_tokens=2, cost_usd=0.01)
        if self.over_budget:
            raise AssertionError("the supervisor should reject the second provider call")
        supervisor.before_tool_call()
        iterations: list[MCEEvolutionIteration] = []
        for index in range(controls["iterations"] + 1):
            root = workspace / f"iteration-{index}"
            root.mkdir()
            if index:
                context = root / "context"
                context.mkdir()
                (context / "context.md").write_text(
                    f"evolved context {index}\n",
                    encoding="utf-8",
                )
                if self.missing_skill != index:
                    skill = root / ".claude" / "skills" / "learning-context"
                    skill.mkdir(parents=True)
                    (skill / "SKILL.md").write_text(
                        f"# Evolved skill {index}\n",
                        encoding="utf-8",
                    )
            iterations.append(
                MCEEvolutionIteration(
                    index=index,
                    artifact_root=root,
                    validation_score=self.score_provider(index),
                )
            )
        return MCEEvolutionRun(iterations=tuple(iterations))

    def evaluate_frozen(self, *, workspace, artifact_root, selected_iteration, test_path,
                        controls, model_fingerprints, supervisor):
        assert not test_path.is_relative_to(workspace)
        assert model_fingerprints == _MODELS
        self.evaluation_calls = getattr(self, "evaluation_calls", 0) + 1
        source = [json.loads(line) for line in test_path.read_text().splitlines()]
        results, errors = [], []
        for index, sample in enumerate(source):
            reservation = supervisor.before_seeded_model_call(
                logical_seed=controls["seed"], stable_subcall_index=index,
                provider_seed=derive_provider_subcall_seed(controls["seed"], index),
                requested_output_tokens=512, input_tokens=10,
            )
            if index:
                supervisor.failed_model_call(reservation_id=reservation)
                errors.append({"sample": sample, "error": "fixture provider timeout",
                               "evaluation": {"metrics": {}}})
            else:
                supervisor.after_model_call(reservation_id=reservation, output_tokens=1)
                results.append({"sample": sample, "evaluation": {"metrics": {"accuracy": 1.0}}})
        return {
            "evaluation": {
                "summary": {"metrics": {"accuracy": 1.0}, "primary_metric": "accuracy",
                            "primary_metric_value": 1.0, "total": len(source),
                            "successful": len(results), "errors": len(errors), "environment": "fixture"},
                "results": results, "errors": errors,
            },
            "output_budget": supervisor.output_budget_receipt(),
            "frozen_artifact_manifest_sha256": _content_hash(mce_driver._artifact_manifest(
                artifact_root, max_files=128, max_bytes=1024 * 1024)),
            "test_oracle_sha256": _sha256_file(test_path),
            "adaptation_performed": False, "generated_code_imported_by_worker": False,
        }

    class assert_outside:
        def __init__(self, workspace: Path, validation_path: Path) -> None:
            self.workspace = workspace
            self.validation_path = validation_path

        def __enter__(self) -> None:
            try:
                self.validation_path.relative_to(self.workspace)
            except ValueError:
                return
            raise AssertionError("validation oracle leaked into generated workspace")

        def __exit__(self, *_args: Any) -> None:
            return None


class DriverClient:
    worker_id = "mce-fixture-v2"

    def __init__(self, driver: MCEWorkerDriver, *, fingerprint_mutator=None) -> None:
        self.driver = driver
        self.fingerprint_mutator = fingerprint_mutator
        self.last_request: dict[str, Any] | None = None
        self.last_response: dict[str, Any] | None = None
        self.calls = 0

    def call(self, payload):
        self.calls += 1
        self.last_request = json.loads(json.dumps(payload))
        response = self.driver.handle(payload)
        self.last_response = json.loads(json.dumps(response))
        return response

    def fingerprint(self):
        identity = json.loads(json.dumps(self.driver.fingerprint()))
        if self.fingerprint_mutator is not None:
            self.fingerprint_mutator(identity)
        return identity


class MCEWorkerDriverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.manifest_path = _write_fixture_checkout(self.source)
        self.store = PersistentArtifactStore(self.root / "persistent-store")
        self.scores = (0.10, 0.45, 0.90, 0.70)
        self.runtime = FixtureRuntime()
        self.driver = self.make_driver(self.runtime)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def make_driver(
        self,
        runtime: FixtureRuntime,
        *,
        source_root: Path | None = None,
    ) -> MCEWorkerDriver:
        runtime.score_provider = lambda index: self.scores[index]

        return MCEWorkerDriver(
            source_root=source_root or self.source,
            runtime=runtime,
            artifact_store=self.store,
            sandbox_attestation=_sandbox(),
            interface_signature=_INTERFACE,
            initial_context_identity=EMPTY_INITIAL_CONTEXT,
            environment_identity=_ENVIRONMENT,
            model_fingerprints=_MODELS,
            manifest_path=self.manifest_path,
            workspace_root=self.root / "workspaces",
            oracle_root=self.root / "oracles",
            fixture_mode=True,
        )

    def recipe(self, client: DriverClient, *, allow_fixture: bool = True) -> MCEOfflineRecipe:
        identity = self.driver.fingerprint()
        return MCEOfflineRecipe(
            client=client,
            interface_signature_sha256=content_hash(_INTERFACE),
            initial_context_sha256=MCE_EMPTY_INITIAL_CONTEXT_SHA256,
            interface_signature=_INTERFACE,
            environment_identity=_ENVIRONMENT,
            license_manifest_allowlist=(identity["license_manifest_sha256"],),
            provenance_manifest_allowlist=(identity["provenance_manifest_sha256"],),
            allow_fixture_runtime=allow_fixture,
            controls=MCEControls(
                iterations=3,
                train_batch_size=1,
                max_model_calls=32,
                max_tool_calls=8,
                max_wall_seconds=600,
                max_artifact_files=16,
                max_artifact_bytes=1024 * 1024,
            ),
        )

    def fit(self, client: DriverClient, *, allow_fixture: bool = True):
        return self.recipe(client, allow_fixture=allow_fixture).fit(
            (
                {
                    "example_id": "train-1",
                    "input": {"question": "train question"},
                    "expected_output": "train answer",
                    "metadata": {"source": "fixture"},
                },
            ),
            validation_examples=(
                {
                    "example_id": "validation-1",
                    "input": {"question": "validation question"},
                    "expected_output": "validation answer",
                    "metadata": {},
                },
            ),
            context=InterventionContext(
                seed=19,
                run_config={"mce_model_fingerprints": _MODELS},
            ),
        )

    def test_fixture_is_explicit_and_rejected_by_production_default(self) -> None:
        client = DriverClient(self.driver)
        with self.assertRaisesRegex(ExternalDependencyError, "fixture workers are disabled"):
            self.fit(client, allow_fixture=False)
        self.assertEqual(client.calls, 0)
        self.assertEqual(self.runtime.calls, 0)

    def test_operator_constructed_driver_serves_host_request_over_jsonl(self) -> None:
        client = DriverClient(self.driver)
        self.fit(client)
        self.assertIsNotNone(client.last_request)
        source = io.StringIO(json.dumps(client.last_request) + "\nnot-json\n")
        sink = io.StringIO()

        self.assertEqual(serve_jsonl(self.driver, source, sink), 0)

        responses = tuple(json.loads(line) for line in sink.getvalue().splitlines())
        self.assertEqual(responses[0]["status"], "ok")
        self.assertEqual(responses[1]["failure_code"], "REQUEST_INVALID")
        self.assertFalse(responses[1]["consumed"])

    def test_evolved_run_persists_opaque_content_addressed_artifact(self) -> None:
        client = DriverClient(self.driver)
        artifact = self.fit(client)

        self.assertEqual(artifact.selected_iteration, 2)
        self.assertEqual(tuple(item.index for item in artifact.iterations), (0, 1, 2, 3))
        self.assertEqual(
            tuple(item.validation_score for item in artifact.iterations),
            self.scores,
        )
        self.assertTrue(artifact.artifact_handle.startswith("urn:agentinputlab:mce:sha256:"))
        stored = self.store.resolve(artifact.artifact_handle)
        self.assertEqual(
            stored["artifact_bundle_sha256"],
            artifact.artifact_bundle_sha256,
        )
        self.assertEqual(
            {item.path for item in artifact.files},
            {
                ".claude/skills/learning-context/SKILL.md",
                "context/context.md",
            },
        )
        self.assertFalse(artifact.host_code_executed)
        self.assertEqual(artifact.provenance_status, "generated_unreviewed")
        self.assertEqual(artifact.usage["model_calls"], 4)
        self.assertEqual(artifact.usage["tool_calls"], 1)
        self.assertEqual(self.runtime.train_records[0]["ground_truth"], "train answer")
        self.assertNotIn("content", to_jsonable(artifact.files[0]))
        self.assertFalse(self.runtime.workspace.exists())
        self.assertFalse(self.runtime.validation_path.exists())

    def test_empty_baseline_artifact_is_a_valid_selected_result(self) -> None:
        self.scores = (0.95, 0.45, 0.20, 0.10)
        artifact = self.fit(DriverClient(self.driver))

        self.assertEqual(artifact.selected_iteration, 0)
        self.assertEqual(artifact.files, ())
        stored = self.store.resolve(artifact.artifact_handle)
        self.assertEqual(stored["files"], [])

    def test_frozen_selection_materializes_without_evolution_or_execution(self) -> None:
        artifact = self.fit(DriverClient(self.driver))
        workspace = self.root / "evaluation-workspace"
        workspace.mkdir()
        before = self.runtime.calls
        first = self.store.materialize(artifact.artifact_handle, workspace=workspace)
        second = self.store.materialize(artifact.artifact_handle, workspace=workspace)
        self.assertNotEqual(first, second)
        self.assertEqual(first.parent, workspace.resolve())
        self.assertEqual(self.runtime.calls, before)
        for row in self.store.resolve(artifact.artifact_handle)["files"]:
            self.assertEqual(_sha256_file(first / row["path"]), row["sha256"])
            self.assertEqual((first / row["path"]).read_bytes(), (second / row["path"]).read_bytes())

    def test_frozen_selection_rejects_store_overlap_symlink_and_limits(self) -> None:
        artifact = self.fit(DriverClient(self.driver))
        workspace = self.root / "evaluation-workspace"
        workspace.mkdir()
        link = self.root / "evaluation-link"
        link.symlink_to(workspace, target_is_directory=True)
        for path in (self.store.root, self.root, link):
            with self.subTest(path=path), self.assertRaises(DriverError):
                self.store.materialize(artifact.artifact_handle, workspace=path)
        with self.assertRaises(DriverError):
            self.store.materialize(artifact.artifact_handle, workspace=workspace, max_bytes=1)
        with self.assertRaises(DriverError):
            self.store.materialize(artifact.artifact_handle, workspace=workspace, max_files=True)
        self.assertEqual(tuple(workspace.iterdir()), ())

    def test_frozen_selection_rejects_modified_payload_without_partial_copy(self) -> None:
        artifact = self.fit(DriverClient(self.driver))
        workspace = self.root / "evaluation-workspace"
        workspace.mkdir()
        stored = self.store.objects / artifact.artifact_bundle_sha256 / "payload"
        (stored / "context/context.md").write_text("changed", encoding="utf-8")
        with self.assertRaises(DriverError):
            self.store.materialize(artifact.artifact_handle, workspace=workspace)
        self.assertEqual(tuple(workspace.iterdir()), ())

    def final_evaluator(self, client):
        artifact = self.fit(client)
        artifact = MCEContextArtifact.from_mapping(json.loads(json.dumps(artifact.as_dict())))
        audit = validate_mce_split_isolation(
            (MCEExample("train-1", {"question": "train question"}, "train answer", "train", {"source": "fixture"}),),
            (MCEExample("validation-1", {"question": "validation question"}, "validation answer", "validation", {}),),
        )
        return MCEFinalArtifactEvaluator(
            artifact=artifact, split_audit=audit, interface_signature=_INTERFACE,
            client=client, controls=MCEFinalControls(max_wall_seconds=600), allow_fixture_runtime=True,
        )

    def test_final_route_retains_failure_denominator_costs_and_never_refits(self):
        client = DriverClient(self.driver)
        evaluator = self.final_evaluator(client)
        examples = tuple(MCEFinalExample(f"test-{i}", {"question": f"heldout {i}"}, "oracle") for i in range(2))
        result = evaluator.evaluate(examples, context=InterventionContext(seed=17, run_config={"mce_model_fingerprints": _MODELS}))
        self.assertEqual(self.runtime.calls, 1)
        self.assertEqual(self.runtime.evaluation_calls, 1)
        self.assertEqual(result.native_summary["total"], 2)
        self.assertEqual(result.native_summary["primary_metric_value"], 1.0)
        self.assertEqual(result.all_positions_primary_metric, 0.5)
        self.assertEqual(result.rows[1]["status"], "failed")
        self.assertEqual(result.rows[1]["metrics"], {})
        self.assertEqual(result.usage["model_calls"], 2)
        self.assertEqual(result.output_budget["failed_unknown_output_reservations"], {"2": 512})
        evidence = self.root / "oracles" / result.evidence_id
        self.assertTrue((evidence / "native-evaluation.json").is_file())
        self.assertTrue((evidence / "terminal.json").is_file())
        with self.assertRaisesRegex(ExternalDependencyError, "ATTEMPT_ALREADY_EXISTS"):
            evaluator.evaluate(examples, context=InterventionContext(seed=17, run_config={"mce_model_fingerprints": _MODELS}))
        self.assertEqual(self.runtime.evaluation_calls, 1)

    def test_final_overlap_and_model_change_rejected_before_dispatch(self):
        client = DriverClient(self.driver)
        evaluator = self.final_evaluator(client)
        for example in (
            MCEFinalExample("train-1", {"question": "new heldout"}),
            MCEFinalExample("new-id", {"question": "train question"}),
        ):
            with self.subTest(example=example), self.assertRaises(ValueError):
                evaluator.evaluate((example,), context=InterventionContext(seed=17, run_config={"mce_model_fingerprints": _MODELS}))
        self.assertEqual(client.calls, 1)

    def test_every_evolved_iteration_requires_skill(self) -> None:
        runtime = FixtureRuntime(missing_skill=2)
        driver = self.make_driver(runtime)
        client = DriverClient(driver)
        with self.assertRaisesRegex(ExternalDependencyError, "ARTIFACT_INVALID"):
            self.fit(client)
        self.assertEqual(client.last_response["failure_code"], "ARTIFACT_INVALID")

    def test_provider_budget_is_enforced_before_second_model_call(self) -> None:
        runtime = FixtureRuntime(over_budget=True)
        driver = self.make_driver(runtime)
        client = DriverClient(driver)
        recipe = self.recipe(client)
        recipe.controls = MCEControls(
            iterations=1,
            train_batch_size=1,
            max_model_calls=1,
            max_tool_calls=1,
            max_wall_seconds=600,
            max_artifact_files=16,
            max_artifact_bytes=1024 * 1024,
        )
        with self.assertRaisesRegex(ExternalDependencyError, "BUDGET_EXCEEDED"):
            recipe.fit(
                (
                    {
                        "example_id": "train-1",
                        "input": {"question": "train"},
                        "expected_output": "answer",
                    },
                ),
                validation_examples=(
                    {
                        "example_id": "validation-1",
                        "input": {"question": "validation"},
                        "expected_output": "answer",
                    },
                ),
                context=InterventionContext(
                    seed=19,
                    run_config={"mce_model_fingerprints": _MODELS},
                ),
            )
        self.assertEqual(runtime.provider_calls, 1)
        self.assertEqual(client.last_response["usage"]["model_calls"], 1)

    def test_provider_subcall_seed_must_match_attested_derivation(self) -> None:
        supervisor = TrustedProviderSupervisor(
            model_calls=2,
            tool_calls=1,
            wall_seconds=60,
        )
        expected_seed = derive_provider_subcall_seed(19, 0)
        self.assertGreaterEqual(expected_seed, 0)
        self.assertLess(expected_seed, 1 << 63)
        with self.assertRaisesRegex(DriverError, "seed does not match"):
            supervisor.before_seeded_model_call(
                logical_seed=19,
                stable_subcall_index=0,
                provider_seed=expected_seed ^ 1,
            )
        self.assertEqual(supervisor.usage()["model_calls"], 0)
        supervisor.before_seeded_model_call(
            logical_seed=19,
            stable_subcall_index=0,
            provider_seed=expected_seed,
        )
        self.assertEqual(supervisor.usage()["model_calls"], 1)
        supervisor.verify_all_model_calls_seeded()
        self.assertNotEqual(
            derive_provider_subcall_seed(19, 1),
            derive_provider_subcall_seed(20, 0),
        )

        bypassed = TrustedProviderSupervisor(
            model_calls=1,
            tool_calls=1,
            wall_seconds=60,
        )
        bypassed.before_model_call()
        with self.assertRaisesRegex(DriverError, "unseeded"):
            bypassed.verify_all_model_calls_seeded()

    def test_source_import_closure_tampering_fails_before_runtime(self) -> None:
        (self.source / "mce" / "main.py").write_text("tampered\n", encoding="utf-8")
        with self.assertRaisesRegex(ExternalDependencyError, "SOURCE_MISMATCH"):
            self.fit(DriverClient(self.driver))
        self.assertEqual(self.runtime.calls, 0)

    def test_host_recomputes_license_and_provenance_manifests(self) -> None:
        for key in ("license_manifest", "provenance_manifest"):
            with self.subTest(key=key):

                def mutate(identity, field=key):
                    identity[field]["tampered"] = True

                with self.assertRaisesRegex(
                    ExternalDependencyError,
                    "does not match its manifest body",
                ):
                    self.fit(DriverClient(self.driver, fingerprint_mutator=mutate))
        self.assertEqual(self.runtime.calls, 0)

    def test_host_rejects_oracle_boundary_attestation_tampering(self) -> None:
        for key, value in (
            ("oracle_mount_mode", "workspace_visible"),
            ("generated_workspace_oracle_visible", True),
            ("provider_guard_mode", "post_call_accounting"),
        ):
            with self.subTest(key=key):

                def mutate(identity, field=key, replacement=value):
                    identity["sandbox_attestation"][field] = replacement

                with self.assertRaisesRegex(
                    ExternalDependencyError,
                    "sandbox attestation violates policy",
                ):
                    self.fit(DriverClient(self.driver, fingerprint_mutator=mutate))
        self.assertEqual(self.runtime.calls, 0)

    def test_full_example_hash_detects_label_tampering(self) -> None:
        client = DriverClient(self.driver)
        self.fit(client)
        request = client.last_request
        request["payload"]["validation_examples"][0]["expected_output"] = "tampered"
        request["request_id"] = content_hash(
            {key: value for key, value in request.items() if key != "request_id"}
        )

        response = self.driver.handle(request)

        self.assertEqual(response["status"], "error")
        self.assertEqual(response["failure_code"], "SPLIT_IDENTITY_MISMATCH")

    def test_request_environment_body_is_bound(self) -> None:
        client = DriverClient(self.driver)
        self.fit(client)
        request = client.last_request
        request["payload"]["environment_identity"]["revision"] = "tampered"
        request["request_id"] = content_hash(
            {key: value for key, value in request.items() if key != "request_id"}
        )

        response = self.driver.handle(request)

        self.assertEqual(response["failure_code"], "IDENTITY_MISMATCH")

    def test_worker_rejects_model_identity_extension_fields(self) -> None:
        models = json.loads(json.dumps(_MODELS))
        models["base"]["api_key"] = "must-not-enter-worker-identity"
        with self.assertRaisesRegex(DriverError, "identity fields drifted"):
            _validate_model_fingerprints(models)

    def test_iteration_rejects_symlinked_artifact_root(self) -> None:
        artifact_root = self.root / "artifact-root"
        artifact_root.mkdir()
        linked_root = self.root / "linked-artifact-root"
        linked_root.symlink_to(artifact_root, target_is_directory=True)
        with self.assertRaisesRegex(DriverError, "may not be a symlink"):
            MCEEvolutionIteration(
                index=0,
                artifact_root=linked_root,
                validation_score=0.0,
            )

    def test_driver_rejects_symlinked_source_root(self) -> None:
        linked_source = self.root / "linked-source"
        linked_source.symlink_to(self.source, target_is_directory=True)
        with self.assertRaisesRegex(DriverError, "may not be a symlink"):
            self.make_driver(FixtureRuntime(), source_root=linked_source)


class PinnedSmokeManager:
    def __init__(self) -> None:
        self.supervisor: TrustedProviderSupervisor | None = None
        self.events: list[str] = []
        self.logical_seed_events: list[tuple[str, int, int]] = []
        self.provider_seeds: list[int] = []

    def fingerprint(self):
        return {
            "schema": "agentinputlab.mce.isolated-manager.v1",
            "generated_code_execution": "isolated_worker_only",
            "oracle_mount_mode": "supervisor_only_hidden",
            "oracle_process_boundary": "trusted_supervisor_out_of_process",
            "generated_workspace_oracle_visible": False,
            "provider_guard_mode": "provider_transport_pre_call",
            "seed_mode": SEED_MODE,
            "provider_subcall_seed_mode": PROVIDER_SUBCALL_SEED_MODE,
            "validation_scoring_mode": "supervisor_only",
            "run_agent_mode": "remote_only",
            "interface_validation_mode": "remote_static_no_import",
            "batch_evaluation_mode": "trusted_supervisor_no_generated_import",
            "source_commit": UPSTREAM_COMMIT,
            "upstream_main_sha256": UPSTREAM_MAIN_SHA256,
            "semantic_patch_manifest_sha256": SEMANTIC_PATCH_MANIFEST_SHA256,
            "generated_worker_identity_sha256": "a" * 64,
            "oracle_supervisor_identity_sha256": "b" * 64,
            "provider_proxy_identity_sha256": "c" * 64,
        }

    def bind_run(
        self,
        *,
        supervisor,
        model_fingerprints,
        seed,
        seed_mode,
        provider_subcall_seed_mode,
        max_wall_seconds,
        semantic_patch_manifest,
    ):
        self.supervisor = supervisor
        self.events.append("bind")
        self.assert_seed_modes(seed_mode, provider_subcall_seed_mode)
        return {
            "schema": "agentinputlab.mce.manager-binding.v1",
            "semantic_patch_manifest_sha256": content_hash(semantic_patch_manifest),
            "model_fingerprints_sha256": content_hash(model_fingerprints),
            "seed": seed,
            "seed_mode": SEED_MODE,
            "provider_subcall_seed_mode": PROVIDER_SUBCALL_SEED_MODE,
            "max_wall_seconds": max_wall_seconds,
            "provider_guard_installed": True,
            "provider_pre_call_reservation": True,
        }

    async def run_agent(self, **kwargs):
        assert self.supervisor is not None
        role = kwargs["role"]
        self.assert_seed_modes(
            kwargs["seed_mode"],
            kwargs["provider_subcall_seed_mode"],
        )
        logical_call_index = kwargs["logical_call_index"]
        logical_seed = kwargs["seed"]
        provider_seed = derive_provider_subcall_seed(logical_seed, 0)
        self.logical_seed_events.append((role, logical_call_index, logical_seed))
        self.provider_seeds.append(provider_seed)
        self.supervisor.before_seeded_model_call(
            logical_seed=logical_seed,
            stable_subcall_index=0,
            provider_seed=provider_seed,
            input_tokens=2,
        )
        self.supervisor.after_model_call(output_tokens=1, cost_usd=0.01)
        self.supervisor.before_tool_call()
        self.events.append(role)
        iter_dir = kwargs["iter_dir"]
        if role == "meta":
            skill_dir = iter_dir / ".claude" / "skills" / "learning-context"
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "SKILL.md").write_text("# isolated skill\n", encoding="utf-8")
        else:
            interfaces = iter_dir / "interfaces"
            interfaces.mkdir(parents=True, exist_ok=True)
            (interfaces / "__init__.py").write_text(
                "from .get_context import get_context\n__all__ = ['get_context']\n",
                encoding="utf-8",
            )
            (interfaces / "get_context.py").write_text(
                "def get_context(symptoms):\n    return ''\n",
                encoding="utf-8",
            )
            context = iter_dir / "context"
            context.mkdir(parents=True, exist_ok=True)
            (context / "knowledge.md").write_text("isolated context\n", encoding="utf-8")
        return {"success": True}

    def validate_interfaces_static(self, **kwargs):
        assert self.supervisor is not None
        self.supervisor.before_tool_call()
        self.events.append("static-validate")
        signatures = kwargs["interface_signatures"]
        artifact_sha256 = _content_hash(_interface_source_manifest(kwargs["iter_dir"]))
        body = {
            "schema": "agentinputlab.mce.remote-interface-handle.v1",
            "semantic_patch_manifest_sha256": SEMANTIC_PATCH_MANIFEST_SHA256,
            "interface_signatures_sha256": _content_hash(signatures),
            "artifact_manifest_sha256": artifact_sha256,
            "interface_names": tuple(item["name"] for item in signatures),
        }
        return {**body, "validation_receipt_sha256": _content_hash(body)}

    async def batch_evaluate(self, **kwargs):
        assert self.supervisor is not None
        self.assert_seed_modes(
            kwargs["seed_mode"],
            kwargs["provider_subcall_seed_mode"],
        )
        logical_call_index = kwargs["logical_call_index"]
        logical_seed = kwargs["seed"]
        provider_seed = derive_provider_subcall_seed(logical_seed, 0)
        self.logical_seed_events.append(("batch_evaluate", logical_call_index, logical_seed))
        self.provider_seeds.append(provider_seed)
        self.supervisor.before_seeded_model_call(
            logical_seed=logical_seed,
            stable_subcall_index=0,
            provider_seed=provider_seed,
            input_tokens=3,
        )
        self.supervisor.after_model_call(output_tokens=1, cost_usd=0.01)
        self.events.append("trusted-evaluate")
        samples = kwargs["samples"]
        score = 0.8 if kwargs["interface_handle"] else 0.2
        results = [
            {
                "sample": sample,
                "evaluation": {
                    "feedback": "trusted",
                    "ground_truth": sample.get("ground_truth"),
                    "metrics": {"accuracy": score},
                    "trajectory": [],
                },
            }
            for sample in samples
        ]
        return {
            "summary": {
                "metrics": {"accuracy": score},
                "primary_metric": "accuracy",
                "primary_metric_value": score,
                "total": len(samples),
                "successful": len(samples),
                "errors": 0,
                "environment": kwargs["environment_name"],
            },
            "errors": [],
            "results": results,
        }

    def usage_receipt(self):
        assert self.supervisor is not None
        return {
            "schema": "agentinputlab.mce.usage-receipt.v1",
            "semantic_patch_manifest_sha256": SEMANTIC_PATCH_MANIFEST_SHA256,
            "usage": self.supervisor.usage(),
        }

    @staticmethod
    def assert_seed_modes(seed_mode, provider_subcall_seed_mode):
        assert seed_mode == SEED_MODE
        assert provider_subcall_seed_mode == PROVIDER_SUBCALL_SEED_MODE


class MCETokenBudgetTests(unittest.TestCase):
    def reserve(self, supervisor, index=0, cap=512):
        return supervisor.before_seeded_model_call(
            logical_seed=17, stable_subcall_index=index,
            provider_seed=derive_provider_subcall_seed(17, index),
            requested_output_tokens=cap, input_tokens=7,
        )

    def test_reserves_actual_request_caps_and_matches_out_of_order_responses(self):
        meter = TrustedEvaluationSupervisor(output_tokens=600, output_tokens_per_call=512)
        first = self.reserve(meter, cap=500)
        second = self.reserve(meter, 1, cap=100)
        with self.assertRaises(DriverError):
            self.reserve(meter, 2, cap=1)
        self.assertEqual(meter.usage()["model_calls"], 2)
        meter.after_model_call(reservation_id=second, output_tokens=20)
        third = self.reserve(meter, 2, cap=80)
        meter.after_model_call(reservation_id=first, output_tokens=450)
        meter.after_model_call(reservation_id=third, output_tokens=75)
        self.assertEqual(meter.usage()["output_tokens"], 545)
        self.assertEqual(meter.usage()["input_tokens"], 21)
        self.assertEqual(meter.output_budget_receipt()["pending_reservations"], {})
        meter.verify_all_model_calls_seeded()
        with self.assertRaises(DriverError):
            meter.after_model_call(reservation_id=first, output_tokens=1)

    def test_rejects_per_call_excess_unseeded_and_bad_seeds_before_counting(self):
        meter = TrustedEvaluationSupervisor()
        with self.assertRaises(DriverError):
            self.reserve(meter, cap=513)
        with self.assertRaises(DriverError):
            meter.before_model_call(input_tokens=10)
        with self.assertRaises(DriverError):
            meter.before_seeded_model_call(
                logical_seed=17, stable_subcall_index=0, provider_seed=18,
                requested_output_tokens=512,
            )
        self.assertEqual(meter.usage()["model_calls"], 0)
        self.reserve(meter)

    def test_failed_unknown_usage_quarantines_budget_and_is_not_zero(self):
        meter = TrustedEvaluationSupervisor(output_tokens=512)
        reservation = self.reserve(meter)
        meter.failed_model_call(reservation_id=reservation)
        with self.assertRaises(DriverError):
            self.reserve(meter, 1, cap=1)
        with self.assertRaises(DriverError):
            meter.after_model_call(reservation_id=reservation, output_tokens=0)
        receipt = meter.output_budget_receipt()
        self.assertEqual(receipt["failed_unknown_output_reservations"], {"1": 512})
        self.assertEqual(receipt["provider_responses"], 0)
        self.assertEqual(meter.usage()["model_calls"], 1)
        self.assertEqual(meter.usage()["unpriced_model_calls"], 1)

    def test_measured_provider_overrun_cost_is_preserved_before_rejection(self):
        meter = TrustedEvaluationSupervisor()
        reservation = self.reserve(meter)
        with self.assertRaises(DriverError) as error:
            meter.after_model_call(reservation_id=reservation, output_tokens=513, cost_usd=0.02)
        self.assertEqual(error.exception.code, "BUDGET_EXCEEDED")
        self.assertEqual(error.exception.usage["output_tokens"], 513)
        self.assertEqual(error.exception.usage["cost_usd"], 0.02)
        self.assertEqual(meter.output_budget_receipt()["provider_responses"], 1)

    def test_call_limit_is_independent_of_unused_token_budget(self):
        meter = TrustedEvaluationSupervisor(model_calls=1)
        reservation = self.reserve(meter)
        meter.after_model_call(reservation_id=reservation, output_tokens=0)
        with self.assertRaises(DriverError):
            self.reserve(meter, 1)
        self.assertEqual(meter.usage()["model_calls"], 1)

    def test_rejects_boolean_caps_and_invalid_response_without_losing_reservation(self):
        for value in (True, 0, -1, 1.5):
            with self.subTest(value=value), self.assertRaises(DriverError):
                TrustedEvaluationSupervisor(output_tokens=value)
        meter = TrustedEvaluationSupervisor()
        reservation = self.reserve(meter)
        with self.assertRaises(DriverError):
            meter.after_model_call(reservation_id=reservation, output_tokens=-1)
        meter.after_model_call(reservation_id=reservation, output_tokens=10)
        self.assertEqual(meter.usage()["output_tokens"], 10)

    def test_concurrent_provider_reservations_cannot_exceed_eight_calls(self):
        meter = TrustedEvaluationSupervisor()
        def reserve(seed):
            try:
                return meter.before_seeded_model_call(
                    logical_seed=seed, stable_subcall_index=0,
                    provider_seed=derive_provider_subcall_seed(seed, 0),
                    requested_output_tokens=512, input_tokens=1,
                )
            except DriverError:
                return None
        with ThreadPoolExecutor(max_workers=16) as pool:
            reservations = tuple(pool.map(reserve, range(32)))
        self.assertEqual(len([value for value in reservations if value is not None]), 8)
        self.assertEqual(meter.usage()["model_calls"], 8)
        self.assertEqual(sum(meter.output_budget_receipt()["pending_reservations"].values()), 4096)

    def test_host_rejects_erased_unknown_usage_and_forged_caps(self):
        meter = TrustedEvaluationSupervisor()
        reservation = self.reserve(meter)
        meter.failed_model_call(reservation_id=reservation)
        controls = {"max_output_tokens": 4096, "max_output_tokens_per_call": 512}
        receipt = meter.output_budget_receipt()
        validate_output_budget(receipt, controls, meter.usage())
        for key, value in (("failed_unknown_output_reservations", {}), ("max_output_tokens_per_call", 1024),
                           ("actual_output_tokens", 512), ("provider_responses", 1)):
            with self.subTest(key=key), self.assertRaises(AlgorithmOutputError):
                validate_output_budget({**receipt, key: value}, controls, meter.usage())

    def test_host_rejects_dropped_failure_rows_and_fabricated_failure_metrics(self):
        rows = [{"example_id": "test", "status": "failed", "metrics": {}, "error": "timeout"}]
        summary = {"total": 1, "successful": 0, "errors": 1, "metrics": {},
                   "primary_metric": "accuracy", "primary_metric_value": 0.0, "environment": "fixture"}
        validate_final_summary(rows, summary, [{"example_id": "test"}])
        for changed in ([], [{**rows[0], "metrics": {"accuracy": 0.0}}]):
            with self.subTest(rows=changed), self.assertRaises(AlgorithmOutputError):
                validate_final_summary(changed, summary, [{"example_id": "test"}])


class FrozenSmokeManager(PinnedSmokeManager):
    """Explicit fixture, not an actual isolated code execution implementation."""

    def final_evaluation_fingerprint(self):
        return final_evaluation_manager_contract()

    async def batch_evaluate(self, **kwargs):
        assert self.supervisor is not None
        reservation = self.supervisor.before_seeded_model_call(
            logical_seed=kwargs["seed"], stable_subcall_index=0,
            provider_seed=derive_provider_subcall_seed(kwargs["seed"], 0),
            requested_output_tokens=512, input_tokens=3,
        )
        self.supervisor.after_model_call(reservation_id=reservation, output_tokens=1, cost_usd=0.01)
        self.events.append("frozen-evaluate")
        if getattr(self, "mutate_artifact", False):
            (kwargs["iter_dir"] / "context" / "knowledge.md").write_text("mutation")
        samples = kwargs["samples"]
        return {
            "summary": {"metrics": {"accuracy": 0.0}, "primary_metric": "accuracy",
                        "primary_metric_value": 0.0, "total": len(samples),
                        "successful": 0, "errors": len(samples),
                        "environment": kwargs["environment_name"]},
            "errors": [{"sample": sample, "error": "fixture real-failure shape"} for sample in samples],
            "results": [],
        }


class MCEOfficialRuntimeTests(unittest.TestCase):
    def test_frozen_evaluation_uses_native_loader_without_training_or_host_imports(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            workspace = root / "workspace"
            workspace.mkdir()
            artifact = workspace / "artifact"
            (artifact / "context").mkdir(parents=True)
            (artifact / "context" / "knowledge.md").write_text("frozen")
            (artifact / "interfaces").mkdir()
            (artifact / "interfaces" / "__init__.py").write_text("# fixture: never imported")
            skill = artifact / ".claude" / "skills" / "learning-context" / "SKILL.md"
            skill.parent.mkdir(parents=True)
            skill.write_text("# frozen")
            source = root / "source"
            (source / "mce").mkdir(parents=True)
            (source / "mce" / "eval.py").write_text("# fixture native source identity")
            test_path = root / "heldout.jsonl"
            test_path.write_text('{"id":"test-1","question":"heldout","answer":"secret"}\n')
            loaded = []

            class Environment:
                def get_interface_signatures(self):
                    return []

                def load_samples(self, *, path, limit, random_sample):
                    loaded.append((path, limit, random_sample))
                    return [json.loads(line) for line in Path(path).read_text().splitlines()][:limit]

            native = types.SimpleNamespace(EnvironmentRegistry=types.SimpleNamespace(get=lambda _: Environment()))

            @contextmanager
            def pinned_modules(_source):
                yield native, object()

            manager = FrozenSmokeManager()
            controls = dict(seed=17, test_limit=1, max_wall_seconds=60,
                            max_artifact_files=128, max_artifact_bytes=1024 * 1024)
            with (
                patch.object(mce_driver, "UPSTREAM_EVAL_SHA256", _sha256_file(source / "mce" / "eval.py")),
                patch.object(mce_driver, "_pinned_upstream_modules", pinned_modules),
            ):
                runtime = MCEOfficialRuntime(source_root=source, environment_name="fixture", sandbox_manager=manager)
                result = runtime.evaluate_frozen(
                    workspace=workspace, artifact_root=artifact, selected_iteration=1,
                    test_path=test_path, controls=controls, model_fingerprints=_MODELS,
                    supervisor=TrustedEvaluationSupervisor(wall_seconds=60),
                )
                self.assertEqual(loaded, [(str(test_path.resolve()), 1, False)])
                self.assertEqual(result["evaluation"]["summary"]["errors"], 1)
                self.assertFalse(result["adaptation_performed"])
                self.assertFalse(result["generated_code_imported_by_worker"])
                self.assertEqual(manager.events, ["bind", "static-validate", "frozen-evaluate"])
                self.assertEqual(result["output_budget"]["actual_output_tokens"], 1)
                self.assertFalse(tuple(workspace.rglob("*heldout*")))
                with self.assertRaisesRegex(DriverError, "overlaps"):
                    runtime.evaluate_frozen(
                        workspace=workspace, artifact_root=artifact, selected_iteration=1,
                        test_path=artifact / "context" / "knowledge.md", controls=controls,
                        model_fingerprints=_MODELS, supervisor=TrustedEvaluationSupervisor(),
                    )

    def test_real_pinned_run_iteration_with_semantic_patch(self) -> None:
        source_text = os.environ.get("AGENTINPUTLAB_MCE_PINNED_SOURCE")
        if not source_text:
            self.skipTest("set AGENTINPUTLAB_MCE_PINNED_SOURCE for the pinned-source smoke")
        source = Path(source_text).resolve()
        self.assertEqual(_sha256_file(source / "mce" / "main.py"), UPSTREAM_MAIN_SHA256)
        managers: list[PinnedSmokeManager] = []
        results = []
        supervisors: list[TrustedProviderSupervisor] = []
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            dotenv = types.ModuleType("dotenv")
            dotenv.load_dotenv = lambda **_kwargs: None
            langchain = types.ModuleType("langchain_openai")
            langchain.OpenAIEmbeddings = object
            claude = types.ModuleType("claude_agent_sdk")
            claude.query = lambda *_args, **_kwargs: None
            claude.ClaudeAgentOptions = object
            claude.ClaudeSDKClient = object

            class AsyncOpenAI:
                def __init__(self, **_kwargs):
                    pass

            openai = types.ModuleType("openai")
            openai.AsyncOpenAI = AsyncOpenAI
            with ExitStack() as stack:
                stack.enter_context(
                    patch.dict(
                        sys.modules,
                        {
                            "dotenv": dotenv,
                            "langchain_openai": langchain,
                            "claude_agent_sdk": claude,
                            "openai": openai,
                        },
                    )
                )
                stack.enter_context(patch.dict(os.environ, {"OPENAI_API_KEY": "fixture"}))
                for run_index in range(2):
                    manager = PinnedSmokeManager()
                    supervisor = TrustedProviderSupervisor(
                        model_calls=16,
                        tool_calls=8,
                        wall_seconds=60,
                    )
                    workspace = root / f"workspace-{run_index}"
                    oracle_root = root / f"oracle-{run_index}"
                    workspace.mkdir()
                    oracle_root.mkdir()
                    train_path = workspace / "train.jsonl"
                    validation_path = oracle_root / "validation.jsonl"
                    record = json.dumps({"question": "fever", "answer": "common cold"}) + "\n"
                    train_path.write_text(record, encoding="utf-8")
                    validation_path.write_text(record, encoding="utf-8")
                    result = MCEOfficialRuntime(
                        source_root=source,
                        environment_name="symptom_diagnosis",
                        sandbox_manager=manager,
                    ).evolve(
                        workspace=workspace,
                        train_path=train_path,
                        validation_path=validation_path,
                        controls={
                            "iterations": 1,
                            "train_batch_size": 1,
                            "train_limit": 1,
                            "validation_limit": 1,
                            "max_wall_seconds": 60,
                            "seed": 19,
                        },
                        model_fingerprints=_MODELS,
                        supervisor=supervisor,
                    )
                    managers.append(manager)
                    results.append(result)
                    supervisors.append(supervisor)
                    frozen_manager = FrozenSmokeManager()
                    frozen_store = PersistentArtifactStore(root / f"frozen-store-{run_index}")
                    selected_root = result.iterations[-1].artifact_root
                    selected_files = mce_driver._artifact_manifest(
                        selected_root, max_files=128, max_bytes=1024 * 1024,
                    )
                    frozen_handle = frozen_store.put(
                        _content_hash({"fixture_selected_files": selected_files}),
                        artifact_root=selected_root, files=selected_files,
                    )
                    frozen_root = frozen_store.materialize(frozen_handle, workspace=workspace)
                    self.assertFalse((frozen_root / "data").exists())
                    frozen = MCEOfficialRuntime(
                        source_root=source, environment_name="symptom_diagnosis",
                        sandbox_manager=frozen_manager,
                    ).evaluate_frozen(
                        workspace=workspace, artifact_root=frozen_root,
                        selected_iteration=1, test_path=validation_path,
                        controls={"seed": 17, "test_limit": 1, "max_wall_seconds": 60,
                                  "max_artifact_files": 128, "max_artifact_bytes": 1024 * 1024},
                        model_fingerprints=_MODELS,
                        supervisor=TrustedEvaluationSupervisor(wall_seconds=60),
                    )
                    self.assertFalse(frozen["adaptation_performed"])
                    self.assertEqual(frozen["evaluation"]["summary"]["total"], 1)
                    self.assertNotIn("meta", frozen_manager.events)
                    self.assertNotIn("base", frozen_manager.events)

        expected_seed_events = [
            ("batch_evaluate", 0, 19),
            ("meta", 1, 20),
            ("batch_evaluate", 2, 21),
            ("base", 3, 22),
            ("batch_evaluate", 4, 23),
        ]
        self.assertEqual(managers[0].logical_seed_events, expected_seed_events)
        self.assertEqual(managers[1].logical_seed_events, expected_seed_events)
        expected_provider_seeds = [
            derive_provider_subcall_seed(logical_seed, 0) for logical_seed in range(19, 24)
        ]
        self.assertEqual(managers[0].provider_seeds, expected_provider_seeds)
        self.assertEqual(managers[1].provider_seeds, managers[0].provider_seeds)
        for result, manager, supervisor in zip(results, managers, supervisors, strict=True):
            self.assertEqual(tuple(item.index for item in result.iterations), (0, 1))
            self.assertEqual(
                tuple(item.validation_score for item in result.iterations),
                (0.2, 0.8),
            )
            self.assertIn("meta", manager.events)
            self.assertIn("base", manager.events)
            self.assertGreaterEqual(manager.events.count("trusted-evaluate"), 3)
            self.assertGreaterEqual(supervisor.usage()["model_calls"], 5)

    def test_runtime_rejects_sys_modules_substitution_and_leaky_fingerprint(self) -> None:
        class LeakySandboxManager(PinnedSmokeManager):
            def fingerprint(self):
                value = super().fingerprint()
                value["generated_workspace_oracle_visible"] = True
                return value

        fake_main = types.ModuleType("mce.main")
        fake_main.run_iteration = lambda **_kwargs: None
        with (
            patch.dict(sys.modules, {"mce.main": fake_main}),
            self.assertRaises(DriverError) as caught,
        ):
            MCEOfficialRuntime(
                source_root=Path.cwd(),
                environment_name="fixture",
                sandbox_manager=LeakySandboxManager(),
            )

        self.assertEqual(caught.exception.code, "ATTESTATION_INVALID")


if __name__ == "__main__":
    unittest.main()
