from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from agentinputlab.contracts import ExecutionBudget, ValidationError
from agentinputlab.integrations.external_workers import (
    DockerWorkerSupervisor,
    LocalSubprocessSupervisor,
    SupervisedWorkerError,
    worker_supervisor_from_environment,
)
from agentinputlab.integrations.external_workers.process import (
    BoundedProcessResult,
    BoundedProcessTimeoutError,
    run_bounded_process,
)
from agentinputlab.observability import (
    AcceleratorSample,
    BudgetLedger,
    EventKind,
    JsonlEventSink,
    ObservedExternalRuntime,
    ObservedRunStatus,
    ResourceExitReason,
    ResourceLimits,
    ResourceReceipt,
    ResourceUsage,
    RunMonitor,
)

_IMAGE = "registry.example/worker@sha256:" + "a" * 64


class QueueRunner:
    def __init__(self, responses: Sequence[BoundedProcessResult | BaseException]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def run(
        self,
        argv: Sequence[str],
        *,
        input_text: str,
        timeout_seconds: float,
        max_input_bytes: int,
        max_stdout_bytes: int,
        max_stderr_bytes: int,
        cwd: str | None,
        environment: Mapping[str, str] | None,
        process_name: str,
    ) -> BoundedProcessResult:
        self.calls.append(
            {
                "argv": tuple(argv),
                "input_text": input_text,
                "timeout_seconds": timeout_seconds,
                "max_input_bytes": max_input_bytes,
                "max_stdout_bytes": max_stdout_bytes,
                "max_stderr_bytes": max_stderr_bytes,
                "cwd": cwd,
                "environment": dict(environment or {}),
                "process_name": process_name,
            }
        )
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class FixtureSampler:
    def __init__(self) -> None:
        self.tokens: list[object] = []

    def start(self) -> object:
        token = object()
        self.tokens.append(token)
        return token

    def stop(self, token: object) -> AcceleratorSample:
        if token is not self.tokens[-1]:
            raise AssertionError("sampler token drifted")
        return AcceleratorSample(
            accelerator_seconds=6,
            accelerator_peak_memory_bytes=16384,
            sampler_identity={"mode": "allocated_device_time", "devices": 2},
        )


class _FakeProcess:
    def __init__(self, pid: int, *, child: bool = False) -> None:
        self.pid = pid
        self._child = child

    def children(self, *, recursive: bool) -> list[_FakeProcess]:
        assert recursive
        return [] if self._child else [_FakeProcess(self.pid + 1, child=True)]

    def cpu_times(self) -> SimpleNamespace:
        return SimpleNamespace(user=0.04, system=0.01)

    def memory_info(self) -> SimpleNamespace:
        return SimpleNamespace(rss=1024 if self._child else 4096)


class BoundedProcessSamplingTests(unittest.TestCase):
    def test_optional_process_tree_and_nvml_sampling_are_per_call(self) -> None:
        observed_root: dict[str, int] = {}

        def process_factory(pid: int) -> _FakeProcess:
            observed_root["pid"] = pid
            return _FakeProcess(pid)

        psutil = SimpleNamespace(__version__="fixture", Process=process_factory)

        def gpu_processes(_handle: object) -> list[SimpleNamespace]:
            return [
                SimpleNamespace(
                    pid=observed_root["pid"],
                    usedGpuMemory=8192,
                )
            ]

        nvml = SimpleNamespace(
            __version__="fixture",
            nvmlInit=lambda: None,
            nvmlShutdown=lambda: None,
            nvmlDeviceGetCount=lambda: 1,
            nvmlDeviceGetHandleByIndex=lambda index: f"gpu-{index}",
            nvmlDeviceGetComputeRunningProcesses_v3=gpu_processes,
            nvmlDeviceGetGraphicsRunningProcesses_v3=lambda _handle: [],
        )
        code = "import json,time; time.sleep(.08); print(json.dumps({'ok': True}))"
        with patch.dict(sys.modules, {"psutil": psutil, "pynvml": nvml}):
            result = run_bounded_process(
                (sys.executable, "-c", code),
                input_text="",
                timeout_seconds=1,
                max_input_bytes=1,
                max_stdout_bytes=4096,
                max_stderr_bytes=4096,
                cwd=None,
                environment=None,
                process_name="resource fixture",
            )

        self.assertGreaterEqual(result.cpu_seconds or 0, 0.1)
        self.assertEqual(result.peak_rss_bytes, 5120)
        self.assertGreater(result.accelerator_seconds or 0, 0)
        self.assertEqual(result.accelerator_peak_memory_bytes, 8192)
        self.assertEqual(
            result.measurement_identity["process_tree"]["status"],
            "available",
        )
        self.assertEqual(
            result.measurement_identity["accelerator"]["status"],
            "available",
        )

    def test_missing_optional_dependencies_are_unavailable_not_zero(self) -> None:
        code = "import json; print(json.dumps({'ok': True}))"

        def unavailable(name: str) -> tuple[None, str]:
            assert name in {"psutil", "pynvml"}
            return None, "dependency_missing"

        with patch(
            "agentinputlab.integrations.external_workers.process._optional_module",
            side_effect=unavailable,
        ):
            result = run_bounded_process(
                (sys.executable, "-c", code),
                input_text="",
                timeout_seconds=1,
                max_input_bytes=1,
                max_stdout_bytes=4096,
                max_stderr_bytes=4096,
                cwd=None,
                environment=None,
                process_name="resource fixture",
            )

        self.assertIsNone(result.cpu_seconds)
        self.assertIsNone(result.peak_rss_bytes)
        self.assertIsNone(result.accelerator_seconds)
        self.assertIsNone(result.accelerator_peak_memory_bytes)
        self.assertEqual(
            result.measurement_identity["process_tree"]["status"],
            "unavailable",
        )
        self.assertEqual(
            result.measurement_identity["accelerator"]["reason"],
            "dependency_missing",
        )

    def test_hard_timeout_retains_partial_process_tree_observation(self) -> None:
        psutil = SimpleNamespace(
            __version__="fixture",
            Process=lambda pid: _FakeProcess(pid),
        )
        with (
            patch.dict(sys.modules, {"psutil": psutil}),
            patch(
                "agentinputlab.integrations.external_workers.process._optional_module",
                side_effect=lambda name: (
                    (psutil, None) if name == "psutil" else (None, "dependency_missing")
                ),
            ),
            self.assertRaises(BoundedProcessTimeoutError) as raised,
        ):
            run_bounded_process(
                (sys.executable, "-c", "import time; time.sleep(1)"),
                input_text="",
                timeout_seconds=0.03,
                max_input_bytes=1,
                max_stdout_bytes=4096,
                max_stderr_bytes=4096,
                cwd=None,
                environment=None,
                process_name="timeout fixture",
            )

        observation = raised.exception.resource_observation
        self.assertIsNotNone(observation.cpu_seconds)
        self.assertEqual(observation.peak_rss_bytes, 5120)
        self.assertEqual(
            observation.measurement_identity["process_tree"]["status"],
            "available",
        )


class ResourceModelTests(unittest.TestCase):
    def test_limits_are_immutable_and_content_hashed(self) -> None:
        limits = ResourceLimits(
            wall_timeout_seconds=10,
            memory_bytes=512 * 1024 * 1024,
            cpus=2,
            pids_limit=64,
        )
        equivalent = ResourceLimits(
            wall_timeout_seconds=10.0,
            memory_bytes=512 * 1024 * 1024,
            cpus=2.0,
            pids_limit=64,
        )
        changed = ResourceLimits(
            wall_timeout_seconds=11,
            memory_bytes=512 * 1024 * 1024,
            cpus=2,
            pids_limit=64,
        )
        self.assertEqual(limits.limit_hash, equivalent.limit_hash)
        self.assertNotEqual(limits.limit_hash, changed.limit_hash)
        with self.assertRaisesRegex(Exception, "cannot assign"):
            limits.cpus = 4  # type: ignore[misc]

    def test_receipt_freezes_identities_and_keeps_peak_out_of_additive_usage(self) -> None:
        limits = ResourceLimits(wall_timeout_seconds=2)
        supervisor_identity = {"nested": {"version": "v1"}}
        receipt = ResourceReceipt(
            limit_hash=limits.limit_hash,
            wall_seconds=1,
            cpu_seconds=0.5,
            peak_rss_bytes=1234,
            accelerator_seconds=2,
            supervisor_identity=supervisor_identity,
            image_identity={"digest": "sha256:" + "b" * 64},
        )
        supervisor_identity["nested"]["version"] = "mutated"
        self.assertEqual(receipt.supervisor_identity["nested"]["version"], "v1")
        with self.assertRaises(TypeError):
            receipt.supervisor_identity["nested"]["version"] = "x"  # type: ignore[index]
        self.assertFalse(hasattr(ResourceUsage(), "peak_rss_bytes"))
        self.assertIsNone(
            ResourceReceipt(
                limit_hash=limits.limit_hash,
                wall_seconds=0,
                supervisor_identity={"kind": "fixture"},
                image_identity={"kind": "fixture"},
            ).accelerator_seconds
        )
        with self.assertRaisesRegex(ValidationError, "timed_out must match"):
            ResourceReceipt(
                limit_hash=limits.limit_hash,
                wall_seconds=1,
                supervisor_identity={"kind": "fixture"},
                image_identity={"kind": "fixture"},
                timed_out=True,
            )

    def test_unavailable_accelerator_measurement_is_distinct_from_measured_zero(
        self,
    ) -> None:
        unavailable = AcceleratorSample()
        measured_zero = AcceleratorSample(
            accelerator_seconds=0,
            accelerator_peak_memory_bytes=0,
            sampler_identity={"status": "available", "mode": "allocation_time"},
        )

        self.assertIsNone(unavailable.accelerator_seconds)
        self.assertEqual(measured_zero.accelerator_seconds, 0)
        with self.assertRaisesRegex(ValidationError, "unavailable"):
            AcceleratorSample(
                accelerator_seconds=0,
                sampler_identity={"status": "unavailable"},
            )

    def test_monitor_snapshots_and_redacts_resource_receipts(self) -> None:
        limits = ResourceLimits(wall_timeout_seconds=2)
        receipt = ResourceReceipt(
            limit_hash=limits.limit_hash,
            wall_seconds=0.1,
            supervisor_identity={
                "kind": "fixture",
                "api_key": "fixture-private-resource-key",
            },
            image_identity={
                "reference": _IMAGE,
                "authorization": "Bearer abcdefghijklmnop",
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            monitor = RunMonitor(sinks=(JsonlEventSink(path),))
            run_id = monitor.start_run(release_id="fixture", state_hash="state")
            recorded = monitor.record_resource_receipt(
                receipt,
                run_id=run_id,
                component="worker",
            )
            snapshot = monitor.finish_run(ObservedRunStatus.SUCCEEDED, run_id=run_id)
            serialized = path.read_text(encoding="utf-8")
        self.assertEqual(recorded.supervisor_identity["api_key"], "<redacted>")
        self.assertEqual(snapshot.resource_receipts, (recorded,))
        self.assertTrue(
            any(event.kind is EventKind.RESOURCE_RECEIPT_RECORDED for event in snapshot.events)
        )
        self.assertNotIn("sk-private", serialized)
        self.assertNotIn("abcdefghijklmnop", serialized)

    def test_observed_external_runtime_attaches_supervisor_receipt(self) -> None:
        limits = ResourceLimits(wall_timeout_seconds=2)
        receipt = ResourceReceipt(
            limit_hash=limits.limit_hash,
            wall_seconds=0.1,
            supervisor_identity={"kind": "fixture"},
            image_identity={"reference": _IMAGE},
        )

        class Runtime:
            worker_id = "fixture"
            last_resource_receipt = receipt

            def fingerprint(self) -> Mapping[str, Any]:
                return {"worker_id": self.worker_id, "revision": "v1"}

            def call(self, _payload: Mapping[str, Any]) -> Mapping[str, Any]:
                return {
                    "usage": {
                        "model_calls": 0,
                        "retrieval_calls": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                    }
                }

        monitor = RunMonitor()
        run_id = monitor.start_run(release_id="fixture", state_hash="state")
        runtime = ObservedExternalRuntime(
            inner=Runtime(),
            monitor=monitor,
            ledger=BudgetLedger(ExecutionBudget()),
            run_id=run_id,
        )
        runtime.call({})
        self.assertEqual(monitor.snapshot(run_id).resource_receipts, (receipt,))


class LocalSupervisorTests(unittest.TestCase):
    def test_local_supervisor_fails_closed_on_unenforceable_limits(self) -> None:
        runner = QueueRunner(())
        supervisor = LocalSubprocessSupervisor(runner=runner)
        for limits, field_name in (
            (ResourceLimits(memory_bytes=1024), "memory_bytes"),
            (ResourceLimits(cpus=1), "cpus"),
            (ResourceLimits(pids_limit=8), "pids_limit"),
        ):
            with (
                self.subTest(field_name=field_name),
                self.assertRaisesRegex(ValidationError, field_name),
            ):
                supervisor.execute(
                    ("worker",),
                    input_text="{}\n",
                    limits=limits,
                    max_input_bytes=1024,
                    max_stdout_bytes=1024,
                    max_stderr_bytes=1024,
                    cwd=None,
                    environment=None,
                    process_name="fixture",
                )
        self.assertEqual(runner.calls, [])

    def test_local_timeout_retains_receipt_and_accelerator_sample(self) -> None:
        runner = QueueRunner((BoundedProcessTimeoutError("fixture exceeded 0.1s"),))
        limits = ResourceLimits(wall_timeout_seconds=0.1)
        with self.assertRaises(SupervisedWorkerError) as raised:
            LocalSubprocessSupervisor(runner=runner).execute(
                ("worker",),
                input_text="{}\n",
                limits=limits,
                max_input_bytes=1024,
                max_stdout_bytes=1024,
                max_stderr_bytes=1024,
                cwd=None,
                environment=None,
                process_name="fixture",
                accelerator_sampler=FixtureSampler(),
            )
        receipt = raised.exception.resource_receipt
        self.assertTrue(receipt.timed_out)
        self.assertEqual(receipt.exit_reason, ResourceExitReason.TIMED_OUT)
        self.assertEqual(receipt.accelerator_seconds, 6)
        self.assertEqual(receipt.accelerator_peak_memory_bytes, 16384)
        self.assertEqual(receipt.limit_hash, limits.limit_hash)


class DockerSupervisorTests(unittest.TestCase):
    def test_image_must_be_digest_pinned(self) -> None:
        with self.assertRaisesRegex(ValidationError, "pinned"):
            DockerWorkerSupervisor("registry.example/worker:latest")

    def test_environment_factory_requires_explicit_digest_pinned_docker_mode(self) -> None:
        self.assertIsInstance(worker_supervisor_from_environment({}), LocalSubprocessSupervisor)
        with self.assertRaisesRegex(ValidationError, "AILAB_WORKER_IMAGE is required"):
            worker_supervisor_from_environment({"AILAB_WORKER_SUPERVISOR": "docker"})
        supervisor = worker_supervisor_from_environment(
            {
                "AILAB_WORKER_SUPERVISOR": "docker",
                "AILAB_WORKER_IMAGE": _IMAGE,
            }
        )
        self.assertIsInstance(supervisor, DockerWorkerSupervisor)

    def test_locked_worker_maps_attested_host_paths_to_fixed_container_paths(self) -> None:
        runner = QueueRunner(
            (
                BoundedProcessResult(0, '{"ok":true}\\n', ""),
                BoundedProcessResult(
                    0,
                    json.dumps({"OOMKilled": False, "ExitCode": 0}),
                    "",
                ),
                BoundedProcessResult(0, "", ""),
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            checkout = root / "upstream"
            adapter = root / "candidate"
            assets = root / "assets"
            checkout.mkdir()
            (checkout / "src").mkdir()
            adapter.mkdir()
            (adapter / "src").mkdir()
            assets.mkdir()
            entrypoint = adapter / "driver.py"
            entrypoint.write_text("print('fixture')\n", encoding="utf-8")
            model_config = assets / "model.json"
            model_config.write_text("{}\n", encoding="utf-8")
            supervisor = DockerWorkerSupervisor(
                _IMAGE,
                runner=runner,
                asset_root=str(assets),
            )
            result = supervisor.execute_locked_worker(
                (sys.executable, str(entrypoint), "--fixture"),
                checkout=checkout,
                adapter_checkout=adapter,
                entrypoint_relative_path="driver.py",
                input_text="{}\n",
                limits=ResourceLimits(
                    wall_timeout_seconds=30,
                    memory_bytes=1024 * 1024,
                    cpus=1,
                    pids_limit=16,
                ),
                max_input_bytes=1024,
                max_stdout_bytes=1024,
                max_stderr_bytes=1024,
                environment={
                    "PWD": str(checkout),
                    "PYTHONPATH": f"{checkout}:{checkout / 'src'}",
                    "AILAB_MODEL_CONFIG": str(model_config),
                    "LANG": "C",
                },
                process_name="fixture",
            )
        run = runner.calls[0]
        argv = run["argv"]
        serialized = " ".join(argv)
        self.assertIn(f"source={checkout},target=/opt/upstream,readonly", serialized)
        self.assertIn(f"source={adapter},target=/opt/candidate,readonly", serialized)
        self.assertIn(f"source={assets},target=/opt/assets,readonly", serialized)
        self.assertIn("--entrypoint /usr/local/bin/python", serialized)
        image_index = argv.index(_IMAGE)
        runtime_argv = argv[image_index + 1 :]
        self.assertEqual(runtime_argv[:3], ("-I", "-B", "-c"))
        self.assertIn("runpy.run_path", runtime_argv[3])
        self.assertEqual(
            runtime_argv[4:],
            (
                "/opt/upstream",
                "/opt/upstream/src",
                "/opt/candidate",
                "/opt/candidate/driver.py",
                "--fixture",
            ),
        )
        self.assertEqual(run["environment"]["PWD"], "/opt/upstream")
        self.assertEqual(
            run["environment"]["PYTHONPATH"],
            "/opt/upstream:/opt/upstream/src:/opt/candidate:/opt/agentinputlab",
        )
        self.assertEqual(
            run["environment"]["AILAB_MODEL_CONFIG"],
            "/opt/assets/model.json",
        )
        self.assertEqual(result.returncode, 0)

    def test_locked_worker_allows_one_checkout_at_both_fixed_targets(self) -> None:
        runner = QueueRunner(
            (
                BoundedProcessResult(0, '{"ok":true}\\n', ""),
                BoundedProcessResult(
                    0,
                    json.dumps({"OOMKilled": False, "ExitCode": 0}),
                    "",
                ),
                BoundedProcessResult(0, "", ""),
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory).resolve() / "candidate"
            checkout.mkdir()
            (checkout / "src").mkdir()
            entrypoint = checkout / "driver.py"
            entrypoint.write_text("print('fixture')\n", encoding="utf-8")
            result = DockerWorkerSupervisor(_IMAGE, runner=runner).execute_locked_worker(
                (sys.executable, str(entrypoint)),
                checkout=checkout,
                adapter_checkout=checkout,
                entrypoint_relative_path="driver.py",
                input_text="{}\n",
                limits=ResourceLimits(
                    wall_timeout_seconds=30,
                    memory_bytes=1024 * 1024,
                    cpus=1,
                    pids_limit=16,
                ),
                max_input_bytes=1024,
                max_stdout_bytes=1024,
                max_stderr_bytes=1024,
                environment={"PWD": str(checkout), "PYTHONPATH": str(checkout)},
                process_name="fixture",
            )

        serialized = " ".join(runner.calls[0]["argv"])
        self.assertIn(
            f"source={checkout},target=/opt/upstream,readonly",
            serialized,
        )
        self.assertIn(
            f"source={checkout},target=/opt/candidate,readonly",
            serialized,
        )
        self.assertEqual(
            runner.calls[0]["environment"]["PYTHONPATH"],
            "/opt/upstream:/opt/upstream/src:/opt/agentinputlab",
        )
        self.assertEqual(result.returncode, 0)

    def test_distinct_adapter_cannot_shadow_same_named_upstream_module(self) -> None:
        runner = QueueRunner(
            (
                BoundedProcessResult(0, '{"ok":true}\\n', ""),
                BoundedProcessResult(
                    0,
                    json.dumps({"OOMKilled": False, "ExitCode": 0}),
                    "",
                ),
                BoundedProcessResult(0, "", ""),
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            checkout = root / "upstream"
            upstream_src = checkout / "src"
            adapter = root / "candidate"
            adapter_src = adapter / "src"
            upstream_src.mkdir(parents=True)
            adapter_src.mkdir(parents=True)
            (upstream_src / "collision.py").write_text(
                "ORIGIN = 'upstream-src'\n",
                encoding="utf-8",
            )
            (adapter / "collision.py").write_text(
                "ORIGIN = 'candidate-root'\n",
                encoding="utf-8",
            )
            (adapter_src / "collision.py").write_text(
                "ORIGIN = 'candidate-src'\n",
                encoding="utf-8",
            )
            (adapter / "adapter_support.py").write_text(
                "IDENTITY = 'candidate-adapter'\n",
                encoding="utf-8",
            )
            entrypoint = adapter / "driver.py"
            entrypoint.write_text(
                "import json\n"
                "from adapter_support import IDENTITY\n"
                "from collision import ORIGIN\n"
                "print(json.dumps({'adapter': IDENTITY, 'algorithm': ORIGIN}, sort_keys=True))\n",
                encoding="utf-8",
            )
            DockerWorkerSupervisor(_IMAGE, runner=runner).execute_locked_worker(
                (sys.executable, str(entrypoint)),
                checkout=checkout,
                adapter_checkout=adapter,
                entrypoint_relative_path="driver.py",
                input_text="{}\n",
                limits=ResourceLimits(
                    wall_timeout_seconds=30,
                    memory_bytes=1024 * 1024,
                    cpus=1,
                    pids_limit=16,
                ),
                max_input_bytes=1024,
                max_stdout_bytes=1024,
                max_stderr_bytes=1024,
                environment={
                    "PWD": str(checkout),
                    "PYTHONPATH": f"{checkout}:{upstream_src}",
                },
                process_name="fixture",
            )
            docker_argv = runner.calls[0]["argv"]
            image_index = docker_argv.index(_IMAGE)
            runtime_argv = list(docker_argv[image_index + 1 :])
            host_paths = {
                "/opt/upstream": str(checkout),
                "/opt/upstream/src": str(upstream_src),
                "/opt/candidate": str(adapter),
                "/opt/candidate/driver.py": str(entrypoint),
            }
            runtime_argv = [host_paths.get(argument, argument) for argument in runtime_argv]
            process = run_bounded_process(
                (sys.executable, *runtime_argv),
                input_text="",
                timeout_seconds=10,
                max_input_bytes=1,
                max_stdout_bytes=1024,
                max_stderr_bytes=1024,
                cwd=str(checkout),
                environment={"PYTHONPATH": "must_be_ignored"},
                process_name="PYTHONPATH collision fixture",
            )

        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(
            json.loads(process.stdout),
            {
                "adapter": "candidate-adapter",
                "algorithm": "upstream-src",
            },
        )

    def test_locked_worker_rejects_partial_checkout_overlap_in_either_direction(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            parent = root / "parent"
            child = parent / "child"
            parent.mkdir()
            child.mkdir()
            for checkout, adapter in ((parent, child), (child, parent)):
                with self.subTest(checkout=checkout, adapter=adapter):
                    runner = QueueRunner(())
                    entrypoint = adapter / "driver.py"
                    entrypoint.write_text("print('fixture')\n", encoding="utf-8")
                    with self.assertRaisesRegex(
                        ValidationError,
                        "must not partially overlap",
                    ):
                        DockerWorkerSupervisor(
                            _IMAGE,
                            runner=runner,
                        ).execute_locked_worker(
                            (sys.executable, str(entrypoint)),
                            checkout=checkout,
                            adapter_checkout=adapter,
                            entrypoint_relative_path="driver.py",
                            input_text="{}\n",
                            limits=ResourceLimits(
                                wall_timeout_seconds=30,
                                memory_bytes=1024 * 1024,
                                cpus=1,
                                pids_limit=16,
                            ),
                            max_input_bytes=1024,
                            max_stdout_bytes=1024,
                            max_stderr_bytes=1024,
                            environment=None,
                            process_name="fixture",
                        )
                    self.assertEqual(runner.calls, [])

    def test_locked_worker_rejects_asset_overlap_with_either_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            separate = root / "separate"
            separate.mkdir()
            cases = []

            exact = root / "exact"
            exact.mkdir()
            cases.append((exact, exact, separate))

            checkout_parent = root / "checkout-parent"
            asset_child = checkout_parent / "assets"
            asset_child.mkdir(parents=True)
            cases.append((asset_child, checkout_parent, separate))

            asset_parent = root / "asset-parent"
            checkout_child = asset_parent / "checkout"
            checkout_child.mkdir(parents=True)
            cases.append((asset_parent, checkout_child, separate))

            adapter_parent = root / "adapter-parent"
            adapter_asset_child = adapter_parent / "assets"
            adapter_asset_child.mkdir(parents=True)
            cases.append((adapter_asset_child, separate, adapter_parent))

            for asset_root, checkout, adapter in cases:
                with self.subTest(
                    asset_root=asset_root,
                    checkout=checkout,
                    adapter=adapter,
                ):
                    runner = QueueRunner(())
                    entrypoint = adapter / "driver.py"
                    entrypoint.write_text("print('fixture')\n", encoding="utf-8")
                    supervisor = DockerWorkerSupervisor(
                        _IMAGE,
                        runner=runner,
                        asset_root=str(asset_root),
                    )
                    with self.assertRaisesRegex(
                        ValidationError,
                        "asset root must not overlap",
                    ):
                        supervisor.execute_locked_worker(
                            (sys.executable, str(entrypoint)),
                            checkout=checkout,
                            adapter_checkout=adapter,
                            entrypoint_relative_path="driver.py",
                            input_text="{}\n",
                            limits=ResourceLimits(
                                wall_timeout_seconds=30,
                                memory_bytes=1024 * 1024,
                                cpus=1,
                                pids_limit=16,
                            ),
                            max_input_bytes=1024,
                            max_stdout_bytes=1024,
                            max_stderr_bytes=1024,
                            environment=None,
                            process_name="fixture",
                        )
                    self.assertEqual(runner.calls, [])

    def test_command_enforces_isolation_limits_and_cleans_up(self) -> None:
        runner = QueueRunner(
            (
                BoundedProcessResult(0, '{"ok":true}\\n', "", 1.25, 4096),
                BoundedProcessResult(
                    0,
                    json.dumps({"OOMKilled": False, "ExitCode": 0}),
                    "",
                ),
                BoundedProcessResult(0, "", ""),
            )
        )
        limits = ResourceLimits(
            wall_timeout_seconds=30,
            memory_bytes=512 * 1024 * 1024,
            cpus=1.5,
            pids_limit=48,
        )
        result = DockerWorkerSupervisor(_IMAGE, runner=runner).execute(
            ("python", "-m", "worker"),
            input_text="{}\n",
            limits=limits,
            max_input_bytes=1024,
            max_stdout_bytes=1024,
            max_stderr_bytes=1024,
            cwd="/worker",
            environment={"API_TOKEN": "do-not-put-in-argv"},
            process_name="fixture",
            image_identity={"upstream_commit": "abc"},
        )
        run_argv = runner.calls[0]["argv"]
        serialized_argv = " ".join(run_argv)
        self.assertEqual(run_argv[:2], ("docker", "run"))
        self.assertIn("--network none", serialized_argv)
        self.assertIn("--read-only", run_argv)
        self.assertIn("--cap-drop ALL", serialized_argv)
        self.assertIn("--security-opt no-new-privileges", serialized_argv)
        self.assertIn("--pids-limit 48", serialized_argv)
        self.assertIn("--memory 536870912b", serialized_argv)
        self.assertIn("--memory-swap 536870912b", serialized_argv)
        self.assertIn("--cpus 1.5", serialized_argv)
        self.assertIn("--env API_TOKEN", serialized_argv)
        self.assertNotIn("do-not-put-in-argv", serialized_argv)
        name = run_argv[run_argv.index("--name") + 1]
        self.assertEqual(
            runner.calls[-1]["argv"],
            ("docker", "rm", "--force", name),
        )
        self.assertEqual(result.resource_receipt.cpu_seconds, 1.25)
        self.assertEqual(result.resource_receipt.peak_rss_bytes, 4096)
        self.assertEqual(result.resource_receipt.limit_hash, limits.limit_hash)

    def test_docker_client_process_metrics_are_not_mislabeled_as_container_usage(
        self,
    ) -> None:
        runner = QueueRunner(
            (
                BoundedProcessResult(
                    0,
                    '{"ok":true}\\n',
                    "",
                    cpu_seconds=2.0,
                    peak_rss_bytes=8192,
                    measurement_identity={
                        "process_tree": {
                            "status": "available",
                            "scope": "spawned_process_tree",
                        },
                        "accelerator": {
                            "status": "unavailable",
                            "reason": "dependency_missing",
                        },
                    },
                ),
                BoundedProcessResult(
                    0,
                    json.dumps({"OOMKilled": False, "ExitCode": 0}),
                    "",
                ),
                BoundedProcessResult(0, "", ""),
            )
        )
        limits = ResourceLimits(
            wall_timeout_seconds=30,
            memory_bytes=1024,
            cpus=1,
            pids_limit=8,
        )

        result = DockerWorkerSupervisor(_IMAGE, runner=runner).execute(
            ("worker",),
            input_text="",
            limits=limits,
            max_input_bytes=1,
            max_stdout_bytes=1024,
            max_stderr_bytes=1024,
            cwd=None,
            environment=None,
            process_name="fixture",
        )

        self.assertIsNone(result.resource_receipt.cpu_seconds)
        self.assertIsNone(result.resource_receipt.peak_rss_bytes)
        measurement = result.resource_receipt.supervisor_identity["measurement"]
        self.assertEqual(measurement["process_tree"]["status"], "unavailable")
        self.assertIn("docker_client", measurement["process_tree"]["reason"])

    def test_oom_and_timeout_are_classified_and_cleanup_is_finally_enforced(self) -> None:
        limits = ResourceLimits(
            wall_timeout_seconds=1,
            memory_bytes=1024,
            cpus=1,
            pids_limit=8,
        )
        oom_runner = QueueRunner(
            (
                BoundedProcessResult(137, "", ""),
                BoundedProcessResult(
                    0,
                    json.dumps({"OOMKilled": True, "ExitCode": 137}),
                    "",
                ),
                BoundedProcessResult(0, "", ""),
            )
        )
        oom = DockerWorkerSupervisor(_IMAGE, runner=oom_runner).execute(
            ("worker",),
            input_text="",
            limits=limits,
            max_input_bytes=1,
            max_stdout_bytes=10,
            max_stderr_bytes=10,
            cwd=None,
            environment=None,
            process_name="fixture",
        )
        self.assertTrue(oom.resource_receipt.oom_killed)
        self.assertEqual(oom.resource_receipt.exit_reason, ResourceExitReason.OOM_KILLED)

        timeout_runner = QueueRunner(
            (
                BoundedProcessTimeoutError("worker exceeded 1s"),
                BoundedProcessResult(
                    0,
                    json.dumps({"OOMKilled": False, "ExitCode": 137}),
                    "",
                ),
                BoundedProcessResult(0, "", ""),
            )
        )
        with self.assertRaises(SupervisedWorkerError) as raised:
            DockerWorkerSupervisor(_IMAGE, runner=timeout_runner).execute(
                ("worker",),
                input_text="",
                limits=limits,
                max_input_bytes=1,
                max_stdout_bytes=10,
                max_stderr_bytes=10,
                cwd=None,
                environment=None,
                process_name="fixture",
            )
        self.assertTrue(raised.exception.resource_receipt.timed_out)
        self.assertEqual(
            raised.exception.resource_receipt.exit_reason,
            ResourceExitReason.TIMED_OUT,
        )
        self.assertEqual(timeout_runner.calls[-1]["argv"][1:3], ("rm", "--force"))


class LocalLockedWorkerTests(unittest.TestCase):
    def test_distinct_adapter_cannot_shadow_same_named_upstream_module(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            checkout = root / "upstream"
            upstream_src = checkout / "src"
            adapter = root / "candidate"
            entrypoint = adapter / "adapters" / "driver.py"
            upstream_src.mkdir(parents=True)
            entrypoint.parent.mkdir(parents=True)
            (upstream_src / "collision.py").write_text(
                "ORIGIN = 'upstream-src'\n",
                encoding="utf-8",
            )
            (entrypoint.parent / "collision.py").write_text(
                "ORIGIN = 'candidate-script-directory'\n",
                encoding="utf-8",
            )
            (adapter / "adapter_support.py").write_text(
                "IDENTITY = 'candidate-adapter'\n",
                encoding="utf-8",
            )
            entrypoint.write_text(
                "import json\n"
                "from adapter_support import IDENTITY\n"
                "from collision import ORIGIN\n"
                "print(json.dumps({'adapter': IDENTITY, 'algorithm': ORIGIN}, sort_keys=True))\n",
                encoding="utf-8",
            )
            result = LocalSubprocessSupervisor().execute_locked_worker(
                (sys.executable, str(entrypoint)),
                checkout=checkout,
                adapter_checkout=adapter,
                entrypoint_relative_path="adapters/driver.py",
                input_text="",
                limits=ResourceLimits(wall_timeout_seconds=10),
                max_input_bytes=1,
                max_stdout_bytes=1024,
                max_stderr_bytes=1024,
                environment={"PYTHONDONTWRITEBYTECODE": "1"},
                process_name="local locked PYTHONPATH collision fixture",
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            json.loads(result.stdout),
            {
                "adapter": "candidate-adapter",
                "algorithm": "upstream-src",
            },
        )
        launcher = result.resource_receipt.supervisor_identity["locked_python_launcher"]
        self.assertEqual(
            launcher["protocol"],
            "agentinputlab.local-locked-python-launcher.v1",
        )


if __name__ == "__main__":
    unittest.main()
