from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, suppress
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agentinputlab.contracts import ExternalDependencyError, InterventionContext
from agentinputlab.integrations.external_workers.ace import (
    ACE_UPSTREAM_COMMIT,
    ACEPlaybook,
    PlaybookBullet,
)
from agentinputlab.recipes.ace import ACEControls, ACEOfflineRecipe
from workers.ace.driver import (
    UPSTREAM_COMMIT,
    ACEWorkerRuntime,
    CompatibilityPolicy,
    Controls,
    LLMConfig,
    ProviderAttemptCompletions,
    UsageTracker,
    WorkerFailure,
    file_sha256,
    load_compatibility_policy,
    load_operator_data_processor,
    load_upstream,
    serve,
    verify_checkout,
)

_FAKE_ACE = r"""
import json

EVENTS = []
PROVIDER_ATTEMPTS = []


class _RetryableFixtureError(RuntimeError):
    pass


class _Completions:
    def __init__(self, role, fail_first=False):
        self.role = role
        self.fail_first = fail_first
        self.failed = False

    def create(self, **kwargs):
        PROVIDER_ATTEMPTS.append((self.role, kwargs["model"]))
        if self.fail_first and not self.failed:
            self.failed = True
            raise _RetryableFixtureError("fixture retry")
        return object()


class _Chat:
    def __init__(self, role, fail_first=False):
        self.completions = _Completions(role, fail_first=fail_first)


class _Client:
    def __init__(self, role, fail_first=False):
        self.chat = _Chat(role, fail_first=fail_first)


def _info(role, model, prompt_tokens, response_tokens):
    return {
        "role": role,
        "model": model,
        "prompt_num_tokens": prompt_tokens,
        "response_num_tokens": response_tokens,
    }


class _Generator:
    def __init__(self, model):
        self.model = model
        self.api_client = _Client("target", fail_first=True)

    def generate(self, **kwargs):
        EVENTS.append(("target", kwargs["call_id"]))
        while True:
            try:
                self.api_client.chat.completions.create(model=self.model)
            except _RetryableFixtureError:
                continue
            break
        response = json.dumps({"final_answer": "answer", "bullet_ids": []})
        return response, [], _info("generator", self.model, 3, 2)


class _Reflector:
    def __init__(self, model):
        self.model = model
        self.api_client = _Client("reflector")

    def reflect(self, **kwargs):
        EVENTS.append(("reflector", kwargs["call_id"]))
        self.api_client.chat.completions.create(model=self.model)
        response = json.dumps({"reflection": "use evidence", "bullet_tags": []})
        return response, [], _info("reflector", self.model, 4, 1)


class _Curator:
    def __init__(self, model):
        self.model = model
        self.api_client = _Client("curator")

    def curate(self, **kwargs):
        EVENTS.append(("curator", kwargs["call_id"]))
        self.api_client.chat.completions.create(model=self.model)
        bullet_id = f"misc-{kwargs['next_global_id']:05d}"
        updated = kwargs["current_playbook"] + (
            f"\n[{bullet_id}] helpful=0 harmful=0 :: Verify the final answer."
        )
        operations = [
            {
                "type": "ADD",
                "section": "others",
                "content": "Verify the final answer.",
            }
        ]
        return updated, kwargs["next_global_id"] + 1, operations, _info(
            "curator", self.model, 5, 2
        )


class ACE:
    def __init__(
        self,
        api_provider,
        generator_model,
        reflector_model,
        curator_model,
        max_tokens,
        initial_playbook,
        use_bulletpoint_analyzer,
    ):
        assert api_provider == "openai"
        assert max_tokens == 128
        assert use_bulletpoint_analyzer is False
        self.generator = _Generator(generator_model)
        self.reflector = _Reflector(reflector_model)
        self.curator = _Curator(curator_model)
        self.playbook = initial_playbook
        self.best_playbook = initial_playbook
        self.next_global_id = 1

    def run(
        self,
        mode,
        train_samples,
        val_samples,
        test_samples,
        data_processor,
        config,
    ):
        EVENTS.append(("ACE.run", mode))
        print("SENSITIVE_FIXTURE_STDOUT")
        assert mode == "offline"
        assert test_samples is None
        assert train_samples[0]["question"] == "Train question"
        assert train_samples[0]["context"] == "Train context"
        assert train_samples[0]["target"] == "answer"
        assert val_samples[0]["question"] == "Validation question"
        assert config["test_workers"] == 1
        initial_response, _, _ = self.generator.generate(call_id="train_gen_initial")
        self.reflector.reflect(call_id="train_reflect", reasoning_trace=initial_response)
        if "[misc-00005] helpful=2" in self.playbook:
            self.playbook = self.playbook.replace(
                "[misc-00005] helpful=2", "[misc-00005] helpful=3"
            )
        self.playbook, self.next_global_id, _, _ = self.curator.curate(
            call_id="train_curate",
            current_playbook=self.playbook,
            next_global_id=self.next_global_id,
        )
        self.generator.generate(call_id="train_post_curate")
        self.best_playbook = self.playbook
        assert data_processor.answer_is_correct(" Answer ", "answer")
        return {"training_results": {"best_validation_accuracy": 0.75}}
"""

_FAKE_PLAYBOOK_UTILS = r"""
import re


def get_next_global_id(playbook):
    values = [int(value) for value in re.findall(r"-(\d+)\]", playbook)]
    return max(values, default=0) + 1
"""

_FINGERPRINTS = {
    "target": {"provider": "openai", "model": "target-model", "revision": "fixture"},
    "reflector": {
        "provider": "openai",
        "model": "reflector-model",
        "revision": "fixture",
    },
    "curator": {
        "provider": "openai",
        "model": "curator-model",
        "revision": "fixture",
    },
}


def llm_config() -> LLMConfig:
    return LLMConfig.from_json(
        json.dumps(
            {
                "api_provider": "openai",
                "max_tokens_per_call": 128,
                "json_mode": True,
                "roles": {
                    role: {
                        "model": fingerprint["model"],
                        "fingerprint": fingerprint,
                        "input_cost_per_million": 1.0,
                        "output_cost_per_million": 2.0,
                    }
                    for role, fingerprint in _FINGERPRINTS.items()
                },
            }
        )
    )


def write_fake_checkout(root: Path) -> CompatibilityPolicy:
    (root / ".git").mkdir(parents=True)
    (root / ".git" / "HEAD").write_text(UPSTREAM_COMMIT + "\n", encoding="utf-8")
    license_text = "Apache License\nVersion 2.0, January 2004\nfixture\n"
    (root / "LICENSE.txt").write_text(license_text, encoding="utf-8")
    (root / "ace").mkdir()
    (root / "ace" / "__init__.py").write_text("from .ace import ACE\n", encoding="utf-8")
    (root / "ace" / "ace.py").write_text(_FAKE_ACE, encoding="utf-8")
    (root / "playbook_utils.py").write_text(_FAKE_PLAYBOOK_UTILS, encoding="utf-8")
    sources = {
        relative: file_sha256(root / relative)
        for relative in ("ace/__init__.py", "ace/ace.py", "playbook_utils.py")
    }
    return CompatibilityPolicy(
        commit=UPSTREAM_COMMIT,
        license_path="LICENSE.txt",
        license_spdx="Apache-2.0",
        license_sha256=hashlib.sha256(license_text.encode()).hexdigest(),
        source_hashes=sources,
        require_clean_checkout=False,
    )


class RuntimeJSONLClient:
    worker_id = "ace-pinned-fixture"

    def __init__(self, runtime: ACEWorkerRuntime) -> None:
        self.runtime = runtime
        self.last_payload = None
        self.last_response = None

    def call(self, payload):
        self.last_payload = payload
        stdin = io.StringIO(json.dumps(payload) + "\n")
        stdout = io.StringIO()
        serve(self.runtime, stdin, stdout)
        lines = stdout.getvalue().splitlines()
        if len(lines) != 1:
            raise AssertionError(f"worker emitted {len(lines)} JSON lines")
        self.last_response = json.loads(lines[0])
        return self.last_response

    def fingerprint(self):
        return {
            "worker_id": self.worker_id,
            "upstream_commit": UPSTREAM_COMMIT,
            "source": "fake-pinned-checkout",
        }


class ACEWorkerDriverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.checkout = Path(self.temp.name)
        self.policy = write_fake_checkout(self.checkout)
        self.runtime = ACEWorkerRuntime(
            self.checkout,
            llm_config(),
            policy=self.policy,
        )

    def tearDown(self) -> None:
        for name in tuple(sys.modules):
            if (
                name == "ace"
                or name.startswith("ace.")
                or name
                in {
                    "playbook_utils",
                    "llm",
                    "logger",
                    "utils",
                }
            ):
                del sys.modules[name]
        self.temp.cleanup()

    def _fit_with_processor(self, processor):
        runtime = ACEWorkerRuntime(
            self.checkout, llm_config(), policy=self.policy, data_processor=processor
        )
        client = RuntimeJSONLClient(runtime)
        artifact = ACEOfflineRecipe(client=client, controls=ACEControls(max_rounds=1)).fit(
            (
                {
                    "example_id": "train-1",
                    "input": {"question": "Train question", "context": "Train context"},
                    "expected_output": "answer",
                },
            ),
            validation_examples=(
                {
                    "example_id": "validation-1",
                    "input": {"question": "Validation question"},
                    "expected_output": "answer",
                },
            ),
            context=InterventionContext(run_config={"ace_model_fingerprints": _FINGERPRINTS}),
        )
        return runtime, client, artifact

    def test_optional_operator_processor_reaches_unchanged_native_loop(self):
        class Processor:
            def __init__(self):
                self.checked = []
                self.validated = False

            def fingerprint(self):
                return {"processor_id": "fixture-feedback-v1", "tuple_value": (1, 2)}

            def validate_examples(self, train, validation):
                assert train[0]["split"] == "train" and validation[0]["split"] == "validation"
                self.validated = True

            def answer_is_correct(self, prediction, target):
                self.checked.append((prediction, target))
                return prediction.strip().casefold() == target.strip().casefold()

            def evaluate_accuracy(self, predictions, targets):
                return 0.25

        processor = Processor()
        _, client, artifact = self._fit_with_processor(processor)
        self.assertTrue(processor.validated)
        self.assertEqual(processor.checked, [(" Answer ", "answer")])
        self.assertEqual(
            client.last_response["diagnostics"]["operator_task_processor"],
            {"processor_id": "fixture-feedback-v1", "tuple_value": [1, 2]},
        )
        self.assertEqual(artifact.selected_iteration, 0)

    def test_operator_processor_can_reject_before_any_provider_attempt(self):
        class Reject:
            def fingerprint(self):
                return {"processor_id": "fixed-fit-only"}

            def validate_examples(self, train, validation):
                raise ValueError("heldout example not admitted")

            def answer_is_correct(self, *args):
                raise AssertionError("must not score")

            def evaluate_accuracy(self, *args):
                raise AssertionError("must not score")

        with self.assertRaisesRegex(ExternalDependencyError, "INVALID_EXAMPLES"):
            self._fit_with_processor(Reject())
        self.assertFalse(sys.modules["ace.ace"].PROVIDER_ATTEMPTS)

    def test_operator_factory_is_startup_bound_hashed_and_stdout_bounded(self):
        class Processor:
            def fingerprint(self):
                return {"processor_id": "retrieval-fixture"}

            def answer_is_correct(self, *args):
                return False

            def evaluate_accuracy(self, *args):
                return 0.0

        def create(**config):
            self.assertEqual(config, {"metric": "native"})
            print("NOT_JSONL_OUTPUT")
            return Processor()

        module = SimpleNamespace(__file__=__file__, create=create)
        captured = io.StringIO()
        with (
            patch.object(sys, "stdout", captured),
            patch("workers.ace.driver.importlib.import_module", return_value=module),
        ):
            processor = load_operator_data_processor(
                "project_feedback:create", '{"metric":"native"}'
            )
        self.assertEqual(captured.getvalue(), "")
        fingerprint = processor.fingerprint()
        self.assertEqual(
            fingerprint["operator_binding"]["factory_module_sha256"], file_sha256(Path(__file__))
        )
        self.assertGreater(fingerprint["operator_binding"]["factory_stdout_bytes"], 0)
        self.assertNotIn("metric", fingerprint["operator_binding"])
        self.assertFalse(processor.answer_is_correct("x", "y"))

    def test_bad_operator_factory_and_non_processor_are_rejected(self):
        for name in ("x", "x;y:z", "path/file.py:make", "x:thing.attr"):
            with self.subTest(name=name), self.assertRaises(WorkerFailure):
                load_operator_data_processor(name)
        with (
            patch(
                "workers.ace.driver.importlib.import_module",
                return_value=SimpleNamespace(__file__=__file__, create=lambda **kw: object()),
            ),
            self.assertRaises(WorkerFailure),
        ):
            load_operator_data_processor("project_feedback:create")

    def test_bound_clients_disable_hidden_sdk_retries(self) -> None:
        self.runtime.llm_config = LLMConfig.from_json(
            json.dumps(
                {
                    "api_provider": "openai",
                    "max_tokens_per_call": 128,
                    "json_mode": True,
                    "base_url": "http://127.0.0.1:8000/v1",
                    "api_key_env": "ACE_LOCAL_TEST_KEY",
                    "roles": {
                        role: {
                            "model": fp["model"],
                            "fingerprint": fp,
                            "input_cost_per_million": 0,
                            "output_cost_per_million": 0,
                        }
                        for role, fp in _FINGERPRINTS.items()
                    },
                }
            )
        )
        constructor = MagicMock()
        system = SimpleNamespace(
            generator=SimpleNamespace(), reflector=SimpleNamespace(), curator=SimpleNamespace()
        )
        with (
            patch.dict(os.environ, {"ACE_LOCAL_TEST_KEY": "unused-local-fixture"}),
            patch(
                "workers.ace.driver.importlib.import_module",
                return_value=SimpleNamespace(OpenAI=constructor),
            ),
        ):
            self.runtime._bind_openai_compatible_clients(system, seed=29)
        self.assertEqual(constructor.call_count, 3)
        for item in constructor.call_args_list:
            self.assertEqual(item.kwargs["max_retries"], 0)
        for role in ("generator", "reflector", "curator"):
            client = getattr(system, role).api_client
            client.chat.completions.create(model=role, max_tokens=128)
            self.assertEqual(
                constructor.return_value.chat.completions.create.call_args.kwargs["extra_body"],
                {"seed": 29},
            )
            with self.assertRaisesRegex(WorkerFailure, "seed differs"):
                client.chat.completions.create(model=role, max_tokens=128, seed=18)
        self.assertIsNone(self.runtime.llm_config.extra_body)

    def test_fit_seed_must_agree_with_deployment_seed_before_provider_calls(self) -> None:
        client = RuntimeJSONLClient(self.runtime)
        ACEOfflineRecipe(client=client).fit(
            (
                {
                    "example_id": "train",
                    "input": {"question": "Train question", "context": "Train context"},
                    "expected_output": "answer",
                },
            ),
            validation_examples=(
                {
                    "example_id": "val",
                    "input": {"question": "Validation question"},
                    "expected_output": "answer",
                },
            ),
            context=InterventionContext(run_config={"ace_model_fingerprints": _FINGERPRINTS}),
        )
        payload = json.loads(json.dumps(client.last_payload))
        config = self._local_constructor_config()
        attempts = len(sys.modules["ace.ace"].PROVIDER_ATTEMPTS)
        for value in (18, True, 17.0):
            with self.subTest(seed=value):
                self.runtime.llm_config = replace(config, extra_body={"seed": value})
                with self.assertRaisesRegex(WorkerFailure, "deployment seed differs"):
                    self.runtime.run(payload)
                self.assertEqual(len(sys.modules["ace.ace"].PROVIDER_ATTEMPTS), attempts)

    def _local_constructor_config(self) -> LLMConfig:
        return LLMConfig.from_json(
            json.dumps(
                {
                    "api_provider": "openai",
                    "max_tokens_per_call": 128,
                    "json_mode": True,
                    "base_url": "http://127.0.0.1:8000/v1",
                    "api_key_env": "ACE_LOCAL_TEST_KEY",
                    "roles": {
                        role: {
                            "model": fp["model"],
                            "fingerprint": fp,
                            "input_cost_per_million": 0,
                            "output_cost_per_million": 0,
                        }
                        for role, fp in _FINGERPRINTS.items()
                    },
                }
            )
        )

    def test_native_constructor_uses_custom_key_and_restores_absent_defaults(self) -> None:
        self.runtime.llm_config = self._local_constructor_config()
        with patch.dict(os.environ, {"ACE_LOCAL_TEST_KEY": "fixture-local-only"}, clear=True):
            with self.runtime._bound_constructor_environment():
                self.assertEqual(os.environ["OPENAI_API_KEY"], "fixture-local-only")
                self.assertEqual(os.environ["OPENAI_BASE_URL"], "http://127.0.0.1:8000/v1")
            self.assertNotIn("OPENAI_API_KEY", os.environ)
            self.assertNotIn("OPENAI_BASE_URL", os.environ)
            self.assertEqual(os.environ["ACE_LOCAL_TEST_KEY"], "fixture-local-only")

    def test_native_constructor_restores_existing_defaults_on_error(self) -> None:
        self.runtime.llm_config = self._local_constructor_config()
        old = {
            "ACE_LOCAL_TEST_KEY": "fixture-chosen",
            "OPENAI_API_KEY": "fixture-original",
            "OPENAI_BASE_URL": "https://original.invalid/v1",
        }
        with patch.dict(os.environ, old, clear=True):
            with (
                self.assertRaisesRegex(ValueError, "constructor fixture"),
                self.runtime._bound_constructor_environment(),
            ):
                self.assertEqual(os.environ["OPENAI_API_KEY"], "fixture-chosen")
                raise ValueError("constructor fixture")
            self.assertEqual(dict(os.environ), old)

    def test_missing_custom_key_fails_before_native_constructor(self) -> None:
        self.runtime.llm_config = self._local_constructor_config()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "fixture-must-not-be-used"}, clear=True):
            with (
                self.assertRaises(WorkerFailure) as caught,
                self.runtime._bound_constructor_environment(),
            ):
                self.fail("native constructor must not run")
            self.assertEqual(caught.exception.code, "MISSING_API_KEY")
            self.assertNotIn("fixture-must-not-be-used", str(caught.exception))
            self.assertEqual(os.environ["OPENAI_API_KEY"], "fixture-must-not-be-used")

    def test_native_provider_without_explicit_endpoint_does_not_rebind_environment(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "fixture-native"}, clear=True):
            with self.runtime._bound_constructor_environment():
                self.assertEqual(dict(os.environ), {"OPENAI_API_KEY": "fixture-native"})
            self.assertEqual(dict(os.environ), {"OPENAI_API_KEY": "fixture-native"})

    def test_missing_core_native_dependency_has_install_hint(self) -> None:
        for module in ("numpy", "openai", "dotenv", "tiktoken"):
            with (
                self.subTest(module=module),
                patch(
                    "workers.ace.driver.importlib.import_module",
                    side_effect=ModuleNotFoundError("private path must not escape", name=module),
                ),
            ):
                with self.assertRaises(WorkerFailure) as caught:
                    load_upstream(self.checkout)
                self.assertEqual(caught.exception.code, "UPSTREAM_DEPENDENCY_MISSING")
                self.assertIn("agentinputlab[ace]", str(caught.exception))
                self.assertNotIn("private path", str(caught.exception))

    def test_unknown_missing_native_module_is_not_an_extra_install_success(self) -> None:
        with (
            patch(
                "workers.ace.driver.importlib.import_module",
                side_effect=ModuleNotFoundError(
                    "private path must not escape", name="not_a_core_extra"
                ),
            ),
            self.assertRaises(WorkerFailure) as caught,
        ):
            load_upstream(self.checkout)
        self.assertEqual(caught.exception.code, "UPSTREAM_IMPORT_FAILED")
        self.assertNotIn("private path", str(caught.exception))

    def test_failed_jsonl_preserves_consumed_attempts(self) -> None:
        client = RuntimeJSONLClient(self.runtime)
        with self.assertRaisesRegex(ExternalDependencyError, "MODEL_CALL_BUDGET_EXCEEDED"):
            ACEOfflineRecipe(
                client=client, controls=ACEControls(max_rounds=1, max_model_calls=1)
            ).fit(
                (
                    {
                        "example_id": "train-1",
                        "input": {"question": "Train question", "context": "Train context"},
                        "expected_output": "answer",
                    },
                ),
                validation_examples=(
                    {
                        "example_id": "validation-1",
                        "input": {"question": "Validation question"},
                        "expected_output": "answer",
                    },
                ),
                context=InterventionContext(run_config={"ace_model_fingerprints": _FINGERPRINTS}),
            )
        self.assertEqual(client.last_response["usage"]["model_calls"], 1)
        self.assertTrue(client.last_response["consumed"])
        self.assertTrue(client.last_response["failed_attempt_tokens_unknown"])

    def test_provider_usage_is_recorded_before_component_parsing(self) -> None:
        tracker = UsageTracker(
            Controls(max_rounds=1, max_model_calls=3, max_tokens=1000, max_cost_usd=1, seed=0),
            llm_config(),
        )
        response = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=7, completion_tokens=5))
        delegate = SimpleNamespace(create=lambda: response)
        self.assertIs(ProviderAttemptCompletions(delegate, tracker, "target").create(), response)
        usage = tracker.as_dict()
        self.assertEqual(
            (usage["model_calls"], usage["input_tokens"], usage["output_tokens"]), (1, 7, 5)
        )

    def test_native_empty_response_is_typed_and_keeps_physical_usage(self) -> None:
        for flag in (
            "training_marked_incorrect_due_to_empty_response",
            "test_marked_incorrect_due_to_empty_response",
        ):
            with self.subTest(flag=flag):
                tracker = UsageTracker(
                    Controls(max_rounds=1, max_model_calls=3, max_tokens=1000, max_cost_usd=1, seed=0),
                    llm_config(),
                )
                response = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=7, completion_tokens=512))
                delegate = SimpleNamespace(create=lambda response=response: response)
                ProviderAttemptCompletions(delegate, tracker, "target").create()
                info = {flag: True, "error": "TEST_INCORRECT: API returned None content"}
                with self.assertRaises(WorkerFailure) as raised:
                    tracker._after_call("target", ("INCORRECT_DUE_TO_EMPTY_RESPONSE", [], info))
                self.assertEqual(raised.exception.code, "UPSTREAM_EMPTY_RESPONSE")
                usage = tracker.as_dict()
                self.assertEqual((usage["model_calls"], usage["input_tokens"], usage["output_tokens"]), (1, 7, 512))
                self.assertEqual(usage["provider_usage_responses"], 1)

    def test_empty_flag_must_be_explicit_boolean_not_generic_metadata_fallback(self) -> None:
        for flag_value in (None, False, "true", 1):
            with self.subTest(flag_value=flag_value):
                tracker = UsageTracker(
                    Controls(max_rounds=1, max_model_calls=3, max_tokens=1000, max_cost_usd=1, seed=0),
                    llm_config(),
                )
                with self.assertRaises(WorkerFailure) as raised:
                    tracker._after_call("target", ("answer", [], {
                        "test_marked_incorrect_due_to_empty_response": flag_value,
                        "prompt_num_tokens": None,
                    }))
                self.assertEqual(raised.exception.code, "INVALID_SCHEMA")

    def _validation_without_results(self, *, metadata_failure=False, unrelated=False):
        namespace = {}
        name = "different_function" if unrelated else "evaluate_test_set"
        source = f"def {name}():\n    if False:\n        final_results = {{}}\n    return final_results\n"
        exec(compile(source, str(self.checkout / "utils.py"), "exec"), namespace)
        fail = namespace[name]
        module = sys.modules[self.runtime.ace_class.__module__]
        native_generate = module._Generator.generate

        def generate(component, **kwargs):
            response, bullets, info = native_generate(component, **kwargs)
            if metadata_failure:
                info = dict(info, prompt_num_tokens=None)
            return response, bullets, info

        def run(component, **kwargs):
            # Reproduce the pinned native validation catch-and-continue.
            with suppress(WorkerFailure):
                component.generator.generate(call_id="validation")
            return fail()

        client = RuntimeJSONLClient(self.runtime)
        with (
            patch.object(module._Generator, "generate", generate),
            patch.object(self.runtime.ace_class, "run", run),
            self.assertRaises(ExternalDependencyError),
        ):
            ACEOfflineRecipe(client=client, controls=ACEControls(max_rounds=1)).fit(
                (
                    {
                        "example_id": "train-1",
                        "input": {"question": "Train question"},
                        "expected_output": "answer",
                    },
                ),
                validation_examples=(
                    {
                        "example_id": "validation-1",
                        "input": {"question": "Validation question"},
                        "expected_output": "answer",
                    },
                ),
                context=InterventionContext(run_config={"ace_model_fingerprints": _FINGERPRINTS}),
            )
        self.assertEqual(client.last_response["status"], "error")
        self.assertEqual(client.last_response["usage"]["model_calls"], 2)
        self.assertTrue(client.last_response["consumed"])
        self.assertNotIn("iterations", client.last_response)
        return client.last_response

    def test_native_empty_validation_preserves_original_guard_failure_and_usage(self):
        result = self._validation_without_results(metadata_failure=True)
        self.assertEqual(result["failure_code"], "INVALID_SCHEMA")
        self.assertIn("prompt_num_tokens", result["message"])

    def test_native_empty_validation_is_typed_failure_not_zero_or_artifact(self):
        result = self._validation_without_results()
        self.assertEqual(result["failure_code"], "UPSTREAM_VALIDATION_NO_RESULTS")
        self.assertEqual(result["usage"]["tokens"], 5)

    def test_unrelated_unbound_local_is_not_masked_as_empty_validation(self):
        result = self._validation_without_results(unrelated=True)
        self.assertEqual(result["failure_code"], "INTERNAL_WORKER_ERROR")

    def test_recipe_fit_runs_pinned_upstream_fixture_through_jsonl(self) -> None:
        client = RuntimeJSONLClient(self.runtime)
        recipe = ACEOfflineRecipe(
            client=client,
            controls=ACEControls(
                max_rounds=2,
                max_model_calls=10,
                max_tokens=1000,
                max_cost_usd=1.0,
                seed=17,
            ),
        )
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            artifact = recipe.fit(
                (
                    {
                        "example_id": "train-1",
                        "input": {"question": "Train question", "context": "Train context"},
                        "expected_output": "answer",
                    },
                ),
                validation_examples=(
                    {
                        "example_id": "validation-1",
                        "input": {"question": "Validation question"},
                        "expected_output": "answer",
                    },
                ),
                context=InterventionContext(
                    seed=17,
                    run_config={"ace_model_fingerprints": _FINGERPRINTS},
                ),
            )

        self.assertEqual(artifact.upstream_commit, ACE_UPSTREAM_COMMIT)
        self.assertEqual(artifact.selected_iteration, 0)
        self.assertEqual(
            [(bullet.bullet_id, bullet.content) for bullet in artifact.playbook.bullets],
            [("misc-00001", "Verify the final answer.")],
        )
        self.assertEqual(client.last_response["usage"]["model_calls"], 5)
        self.assertEqual(client.last_response["usage"]["tokens"], 22)
        self.assertNotIn("SENSITIVE_FIXTURE_STDOUT", stderr.getvalue())
        self.assertEqual(
            client.last_response["diagnostics"],
            {
                "upstream_stdout_bytes": len("SENSITIVE_FIXTURE_STDOUT\n"),
                "upstream_stdout_sha256": hashlib.sha256(b"SENSITIVE_FIXTURE_STDOUT\n").hexdigest(),
                "content_retained": False,
                "provider_binding": "upstream_native",
            },
        )
        self.assertEqual(
            [role for role, _ in sys.modules["ace.ace"].EVENTS],
            ["ACE.run", "target", "reflector", "curator", "target"],
        )
        self.assertEqual(
            [role for role, _ in sys.modules["ace.ace"].PROVIDER_ATTEMPTS],
            ["target", "target", "reflector", "curator", "target"],
        )

    def test_native_playbook_counters_metadata_and_ids_replay_exactly(self) -> None:
        client = RuntimeJSONLClient(self.runtime)
        initial = ACEPlaybook(
            (
                PlaybookBullet(
                    bullet_id="misc-00005",
                    content="Check evidence.",
                    helpful=2,
                    harmful=1,
                    metadata={"section": "OTHERS", "origin": "fixture"},
                ),
            )
        )
        artifact = ACEOfflineRecipe(
            client=client,
            initial_playbook=initial,
            controls=ACEControls(
                max_rounds=1,
                max_model_calls=10,
                max_tokens=1000,
                max_cost_usd=1.0,
            ),
        ).fit(
            (
                {
                    "example_id": "train-1",
                    "input": {"question": "Train question", "context": "Train context"},
                    "expected_output": "answer",
                },
            ),
            validation_examples=(
                {
                    "example_id": "validation-1",
                    "input": {"question": "Validation question"},
                    "expected_output": "answer",
                },
            ),
            context=InterventionContext(
                run_config={"ace_model_fingerprints": _FINGERPRINTS},
            ),
        )

        first, second = artifact.playbook.bullets
        self.assertEqual((first.bullet_id, first.helpful, first.harmful), ("misc-00005", 3, 1))
        self.assertEqual(first.metadata["origin"], "fixture")
        self.assertEqual(second.bullet_id, "misc-00006")
        operations = client.last_response["iterations"][0]["operations"]
        self.assertEqual([item["operation"] for item in operations], ["edit", "add"])
        self.assertEqual(
            client.last_response["iterations"][0]["candidate_playbook_hash"],
            artifact.playbook.playbook_hash,
        )

    def test_request_id_binds_expected_outputs_and_example_metadata(self) -> None:
        client = RuntimeJSONLClient(self.runtime)
        ACEOfflineRecipe(
            client=client,
            controls=ACEControls(
                max_rounds=1,
                max_model_calls=10,
                max_tokens=1000,
                max_cost_usd=1.0,
            ),
        ).fit(
            (
                {
                    "example_id": "train-1",
                    "input": {"question": "Train question", "context": "Train context"},
                    "expected_output": "answer",
                    "metadata": {"source": "original"},
                },
            ),
            validation_examples=(
                {
                    "example_id": "validation-1",
                    "input": {"question": "Validation question"},
                    "expected_output": "answer",
                },
            ),
            context=InterventionContext(
                run_config={"ace_model_fingerprints": _FINGERPRINTS},
            ),
        )

        mutations = (
            (
                "expected output",
                lambda payload: payload["train_examples"][0].update(
                    {"expected_output": "tampered"}
                ),
            ),
            (
                "metadata",
                lambda payload: payload["train_examples"][0]["metadata"].update(
                    {"source": "tampered"}
                ),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                payload = json.loads(json.dumps(client.last_payload))
                mutate(payload)
                with self.assertRaises(WorkerFailure) as raised:
                    self.runtime.run(payload)
                self.assertEqual(raised.exception.code, "REQUEST_ID_MISMATCH")

    def test_provider_retry_attempts_share_a_hard_pre_call_budget(self) -> None:
        client = RuntimeJSONLClient(self.runtime)
        recipe = ACEOfflineRecipe(
            client=client,
            controls=ACEControls(
                max_rounds=1,
                max_model_calls=4,
                max_tokens=1000,
                max_cost_usd=1.0,
            ),
        )
        with self.assertRaisesRegex(ExternalDependencyError, "MODEL_CALL_BUDGET_EXCEEDED"):
            recipe.fit(
                (
                    {
                        "example_id": "train-1",
                        "input": {"question": "Train question", "context": "Train context"},
                        "expected_output": "answer",
                    },
                ),
                validation_examples=(
                    {
                        "example_id": "validation-1",
                        "input": {"question": "Validation question"},
                        "expected_output": "answer",
                    },
                ),
                context=InterventionContext(
                    run_config={"ace_model_fingerprints": _FINGERPRINTS},
                ),
            )
        self.assertEqual(
            [role for role, _ in sys.modules["ace.ace"].PROVIDER_ATTEMPTS],
            ["target", "target", "reflector", "curator"],
        )

    def test_checkout_hash_license_and_revision_are_fail_closed(self) -> None:
        self.assertEqual(verify_checkout(self.checkout, self.policy), self.checkout.resolve())

        source = self.checkout / "ace" / "ace.py"
        source.write_text(_FAKE_ACE + "\n# tampered\n", encoding="utf-8")
        with self.assertRaisesRegex(WorkerFailure, "source hash mismatch") as raised:
            verify_checkout(self.checkout, self.policy)
        self.assertEqual(raised.exception.code, "UPSTREAM_SOURCE_MISMATCH")

    def test_default_policy_requires_clean_checkout_and_rejects_dirty_status(self) -> None:
        self.assertTrue(load_compatibility_policy().require_clean_checkout)
        result = subprocess.CompletedProcess(
            args=("git", "status"),
            returncode=0,
            stdout="?? untracked.py\n",
            stderr="",
        )
        with (
            patch("workers.ace.driver.subprocess.run", return_value=result) as run,
            self.assertRaises(WorkerFailure) as raised,
        ):
            verify_checkout(
                self.checkout,
                replace(self.policy, require_clean_checkout=True),
            )
        self.assertEqual(raised.exception.code, "DIRTY_CHECKOUT")
        self.assertEqual(run.call_count, 1)
        self.assertNotIn("shell", run.call_args.kwargs)

    def test_upstream_import_restores_dotenv_disable_environment(self) -> None:
        key = "PYTHON_DOTENV_DISABLED"
        originally_present = key in os.environ
        original = os.environ.get(key)
        try:
            os.environ[key] = "original-setting"
            load_upstream(self.checkout)
            self.assertEqual(os.environ[key], "original-setting")

            os.environ.pop(key)
            load_upstream(self.checkout)
            self.assertNotIn(key, os.environ)
        finally:
            if originally_present:
                assert original is not None
                os.environ[key] = original
            else:
                os.environ.pop(key, None)

    def test_appworld_import_is_rejected_before_upstream_loading(self) -> None:
        source = self.checkout / "ace" / "ace.py"
        source.write_text("import appworld\n" + _FAKE_ACE, encoding="utf-8")
        policy = CompatibilityPolicy(
            commit=self.policy.commit,
            license_path=self.policy.license_path,
            license_spdx=self.policy.license_spdx,
            license_sha256=self.policy.license_sha256,
            source_hashes={
                **dict(self.policy.source_hashes),
                "ace/ace.py": file_sha256(source),
            },
            require_clean_checkout=False,
        )
        with self.assertRaisesRegex(WorkerFailure, "refuses to import AppWorld") as raised:
            verify_checkout(self.checkout, policy)
        self.assertEqual(raised.exception.code, "APPWORLD_FORBIDDEN")

    def test_request_cannot_override_runtime_path_or_model_identity(self) -> None:
        client = RuntimeJSONLClient(self.runtime)
        recipe = ACEOfflineRecipe(
            client=client,
            controls=ACEControls(
                max_rounds=1,
                max_model_calls=10,
                max_tokens=1000,
                max_cost_usd=1.0,
            ),
        )
        recipe.fit(
            (
                {
                    "example_id": "train-1",
                    "input": {"question": "Train question", "context": "Train context"},
                    "expected_output": "answer",
                },
            ),
            validation_examples=(
                {
                    "example_id": "validation-1",
                    "input": {"question": "Validation question"},
                    "expected_output": "answer",
                },
            ),
            context=InterventionContext(
                run_config={"ace_model_fingerprints": _FINGERPRINTS},
            ),
        )
        payload = json.loads(json.dumps(client.last_payload))
        payload["controls"]["checkout"] = "/tmp/attacker"
        with self.assertRaises(WorkerFailure) as path_error:
            self.runtime.run(payload)
        self.assertEqual(path_error.exception.code, "INVALID_SCHEMA")

        payload = json.loads(json.dumps(client.last_payload))
        payload["model_fingerprints"]["target"]["model"] = "different-model"
        with self.assertRaises(WorkerFailure) as model_error:
            self.runtime.run(payload)
        self.assertEqual(model_error.exception.code, "MODEL_FINGERPRINT_MISMATCH")

    def test_operator_openai_compatible_endpoint_is_startup_only_and_strict(self) -> None:
        raw = {
            "api_provider": "openai",
            "max_tokens_per_call": 128,
            "json_mode": True,
            "base_url": "http://127.0.0.1:8000/v1/",
            "api_key_env": "OPENAI_API_KEY",
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            "roles": {
                role: {
                    "model": fingerprint["model"],
                    "fingerprint": fingerprint,
                    "input_cost_per_million": 0.0,
                    "output_cost_per_million": 0.0,
                }
                for role, fingerprint in _FINGERPRINTS.items()
            },
        }
        configured = LLMConfig.from_json(json.dumps(raw))
        self.assertEqual(configured.base_url, "http://127.0.0.1:8000/v1")
        self.assertEqual(configured.api_key_env, "OPENAI_API_KEY")
        self.assertEqual(
            configured.extra_body,
            {"chat_template_kwargs": {"enable_thinking": False}},
        )

        for label, mutate in (
            ("missing key environment", lambda value: value.pop("api_key_env")),
            ("embedded credentials", lambda value: value.update(base_url="http://u:p@x/v1")),
            ("query parameters", lambda value: value.update(base_url="https://x/v1?q=1")),
        ):
            with self.subTest(label=label):
                candidate = json.loads(json.dumps(raw))
                mutate(candidate)
                with self.assertRaises(WorkerFailure):
                    LLMConfig.from_json(json.dumps(candidate))

    def test_jsonl_boundary_returns_one_structured_error_line(self) -> None:
        output = io.StringIO()
        serve(self.runtime, io.StringIO('{"request_id":"x","request_id":"y"}\n'), output)
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        response = json.loads(lines[0])
        self.assertEqual(response["status"], "error")
        self.assertEqual(response["failure_code"], "INVALID_JSON")

    @unittest.skipUnless(
        os.environ.get("ACE_UPSTREAM_CHECKOUT"),
        "set ACE_UPSTREAM_CHECKOUT for the pinned real-checkout import smoke",
    )
    def test_real_pinned_checkout_import_smoke(self) -> None:
        checkout = Path(os.environ["ACE_UPSTREAM_CHECKOUT"])
        policy = load_compatibility_policy()
        self.assertEqual(verify_checkout(checkout, policy), checkout.resolve())
        ace_class, playbook_utils = load_upstream(checkout)
        self.assertTrue(callable(ace_class.run))
        self.assertTrue(callable(playbook_utils.get_next_global_id))


if __name__ == "__main__":
    unittest.main()
