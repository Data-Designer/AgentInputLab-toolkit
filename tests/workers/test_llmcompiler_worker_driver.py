from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest import SkipTest

try:
    import pytest
except ModuleNotFoundError as error:
    if error.name != "pytest":
        raise
    raise SkipTest("pytest is required for this test module") from error

from agentinputlab.algorithms.decompose.llm_compiler import (
    LLMCompilerControls,
    LLMCompilerFullBridge,
    LLMCompilerPlannerBridge,
)
from agentinputlab.connectors.tools import ToolCall, ToolResult, ToolSpec
from agentinputlab.contracts import FrozenAgentState, InterventionContext
from workers.llmcompiler import driver

ROOT = Path(__file__).resolve().parents[2]
REAL_CHECKOUT = Path(os.environ.get("AGENTINPUTLAB_LLMCOMPILER_CHECKOUT", "/private/tmp/agentinputlab-upstreams/LLMCompiler"))


def _is_git_checkout(root: Path) -> bool:
    if not root.is_dir():
        return False
    try:
        result = subprocess.run(
            ("git", "-C", str(root), "rev-parse", "--verify", "HEAD"),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return result.returncode == 0


class FingerprintedBackend:
    def __init__(self, replay_manifest_sha256: str) -> None:
        self.replay_manifest_sha256 = replay_manifest_sha256

    def fingerprint(self) -> dict[str, str]:
        return {"replay_manifest_sha256": self.replay_manifest_sha256}


class ManifestToolConnector:
    def __init__(self, tool_manifest_sha256: str) -> None:
        self.tool_manifest_sha256 = tool_manifest_sha256
        self.spec = ToolSpec(
            name="echo",
            description="echo(text: string) -> string",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
            read_only=True,
            network=False,
        )

    def list_tools(self) -> tuple[ToolSpec, ...]:
        return (self.spec,)

    def call(self, request: ToolCall) -> ToolResult:
        raise AssertionError(f"planner component must not execute tool {request.name!r}")

    def fingerprint(self) -> dict[str, str]:
        return {"tool_manifest_sha256": self.tool_manifest_sha256}


class InProcessReplayWorker:
    worker_id = "llmcompiler-replay-e2e"

    def __init__(
        self,
        *,
        upstream: driver.LoadedUpstream,
        model_path: Path,
        tool_path: Path,
    ) -> None:
        self.upstream = upstream
        self.model_path = model_path
        self.tool_path = tool_path
        self.calls = 0

    def call(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError(f"typed call_controlled seam expected, received {payload!r}")

    def call_controlled(self, request: Any) -> dict[str, Any]:
        self.calls += 1
        validated = driver.validate_request(request.as_dict())
        model = driver.load_model_replay(self.model_path, validated)
        tools = driver.load_tool_replay(self.tool_path, validated)
        return asyncio.run(driver.execute_request(validated, self.upstream, model, tools))

    def fingerprint(self) -> dict[str, str]:
        return {
            "worker_id": self.worker_id,
            "upstream_commit": driver.UPSTREAM_COMMIT,
        }


class FakeTool:
    def __init__(self, *, name: str, func: Any, description: str) -> None:
        self.name = name
        self.func = func
        self.description = description


class FakeTask:
    def __init__(
        self,
        idx: int,
        name: str,
        tool: Any,
        args: tuple[Any, ...],
        dependencies: tuple[int, ...] = (),
        *,
        is_join: bool = False,
    ) -> None:
        self.idx = idx
        self.name = name
        self.tool = tool
        self.args = args
        self.dependencies = dependencies
        self.observation = None
        self.is_join = is_join

    async def __call__(self) -> Any:
        return await self.tool(*self.args)


class FakeTaskFetchingUnit:
    def __init__(self) -> None:
        self.tasks: dict[int, FakeTask] = {}
        self.tasks_done: dict[int, asyncio.Event] = {}
        self.remaining_tasks: set[int] = set()
        self.running_tasks: set[asyncio.Task[Any]] = set()

    def _start_task(self, task: FakeTask) -> None:
        running = asyncio.create_task(self._run_task(task))
        self.running_tasks.add(running)
        running.add_done_callback(self.running_tasks.discard)

    def set_tasks(self, tasks: dict[int, FakeTask]) -> None:
        self.tasks.update(tasks)
        self.tasks_done.update({idx: asyncio.Event() for idx in tasks})
        self.remaining_tasks.update(tasks)

    def _all_tasks_done(self) -> bool:
        return all(event.is_set() for event in self.tasks_done.values())

    def _get_all_executable_tasks(self) -> list[int]:
        return [
            idx
            for idx in sorted(self.remaining_tasks)
            if all(
                self.tasks_done[dependency].is_set() for dependency in self.tasks[idx].dependencies
            )
        ]

    async def _run_task(self, task: FakeTask) -> None:
        if not task.is_join:
            task.observation = await task()
        self.tasks_done[task.idx].set()

    async def schedule(self) -> None:
        while not self._all_tasks_done():
            running = self._get_all_executable_tasks()
            for idx in running:
                self.remaining_tasks.remove(idx)
                self._start_task(self.tasks[idx])
            await asyncio.sleep(0)

    async def aschedule(self, task_queue: asyncio.Queue[Any], func: Any) -> None:
        del func
        no_more_tasks = False
        while True:
            if not no_more_tasks:
                task = await task_queue.get()
                if task is None:
                    no_more_tasks = True
                else:
                    self.set_tasks({task.idx: task})
            running = self._get_all_executable_tasks()
            for idx in running:
                self.remaining_tasks.remove(idx)
                self._start_task(self.tasks[idx])
            if no_more_tasks and self._all_tasks_done():
                return
            await asyncio.sleep(0)


def fake_upstream() -> tuple[driver.LoadedUpstream, dict[str, int]]:
    planner_module = ModuleType("fake.planner")
    task_module = ModuleType("fake.task_fetching")
    compiler_module = ModuleType("fake.compiler")
    tools_module = ModuleType("fake.tools")
    logger_module = ModuleType("fake.logger")
    calls = {"planner": 0, "aplan": 0, "task_fetcher": 0, "join": 0, "compiler": 0}

    class FakePlanner:
        def __init__(self, *, llm: Any, tools: list[FakeTool], **_: Any) -> None:
            self.llm = llm
            self.tools = tools

        def _tasks(self, response: str) -> dict[int, FakeTask]:
            if response == "PLAN:first":
                task_id, argument = 1, "first"
            elif response == "PLAN:second":
                task_id, argument = 2, "second"
            else:
                raise AssertionError(f"unexpected fake plan: {response}")
            return {
                task_id: FakeTask(task_id, "echo", self.tools[0].func, (argument,)),
                task_id + 100: FakeTask(
                    task_id + 100,
                    "join",
                    lambda: None,
                    (),
                    (task_id,),
                    is_join=True,
                ),
            }

        async def plan(self, *, inputs: dict[str, Any], is_replan: bool, **_: Any) -> Any:
            calls["planner"] += 1
            prompt = f"planner:{inputs['input']}:{is_replan}"
            return self._tasks(await self.llm.apredict(prompt))

        async def aplan(
            self,
            *,
            inputs: dict[str, Any],
            task_queue: asyncio.Queue[Any],
            is_replan: bool,
            **_: Any,
        ) -> None:
            calls["aplan"] += 1
            prompt = f"planner:{inputs['input']}:{is_replan}"
            tasks = self._tasks(await self.llm.apredict(prompt))
            for task in tasks.values():
                await task_queue.put(task)
            await task_queue.put(None)

    class CountingTaskFetchingUnit(FakeTaskFetchingUnit):
        def __init__(self) -> None:
            calls["task_fetcher"] += 1
            super().__init__()

    task_module.TaskFetchingUnit = CountingTaskFetchingUnit
    planner_module.Planner = FakePlanner
    tools_module.Tool = FakeTool

    class FakeCompiler:
        def __init__(
            self,
            *,
            tools: list[FakeTool],
            planner_llm: Any,
            planner_stream: bool,
            agent_llm: Any,
            max_replans: int,
            **_: Any,
        ) -> None:
            calls["compiler"] += 1
            self.planner = planner_module.Planner(llm=planner_llm, tools=tools)
            self.planner_stream = planner_stream
            self.agent_llm = agent_llm
            self.max_replans = max_replans

        async def join(
            self,
            input_query: str,
            agent_scratchpad: str,
            is_final: bool,
        ) -> tuple[str, str, bool]:
            calls["join"] += 1
            prompt = f"join:{input_query}:{agent_scratchpad}:{is_final}"
            response = await self.agent_llm.agenerate_prompt([SimpleNamespace(text=prompt)])
            raw = response.generations[0][0].text
            action = next(line for line in raw.splitlines() if line.startswith("Action:"))
            answer = action[action.find("(") + 1 : action.rfind(")")]
            return "thought", answer, "Replan" in action and not is_final

        async def _acall(self, inputs: dict[str, Any]) -> dict[str, str]:
            answer = ""
            for index in range(self.max_replans):
                unit = compiler_module.TaskFetchingUnit()
                if self.planner_stream:
                    queue: asyncio.Queue[Any] = asyncio.Queue()
                    planner_job = asyncio.create_task(
                        self.planner.aplan(
                            inputs=inputs,
                            task_queue=queue,
                            is_replan=index > 0,
                        )
                    )
                    await unit.aschedule(queue, lambda _: None)
                    await planner_job
                else:
                    tasks = await self.planner.plan(inputs=inputs, is_replan=index > 0)
                    unit.set_tasks(tasks)
                    await unit.schedule()
                scratchpad = "|".join(
                    str(task.observation) for task in unit.tasks.values() if not task.is_join
                )
                _, answer, is_replan = await self.join(
                    inputs["input"], scratchpad, index == self.max_replans - 1
                )
                if not is_replan:
                    break
                inputs["context"] = "fake previous plan"
            return {"output": answer}

    compiler_module.LLMCompiler = FakeCompiler
    compiler_module.TaskFetchingUnit = CountingTaskFetchingUnit
    logger_module.enable_logging = lambda _: None
    return (
        driver.LoadedUpstream(
            planner=planner_module,
            task_fetching=task_module,
            compiler=compiler_module,
            tools=tools_module,
            logger=logger_module,
        ),
        calls,
    )


def write_json(path: Path, value: Any) -> str:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return driver._sha256_bytes(path.read_bytes())


def prompts() -> dict[str, Any]:
    return {
        "planner_example_prompt": "planner examples",
        "planner_example_prompt_replan": "replanner examples",
        "planner_stop": ["<END_OF_PLAN>"],
        "joinner_prompt": "join prompt",
        "joinner_prompt_final": "final join prompt",
    }


def response_entry(channel: str, prompt: str, response: str) -> dict[str, Any]:
    return {
        "channel": channel,
        "prompt_sha256": driver._sha256_text(prompt),
        "response": response,
        "response_sha256": driver._sha256_text(response),
        "input_tokens": 5,
        "output_tokens": 3,
    }


def tool_manifest(*, cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "agentinputlab.llmcompiler.tool-replay.v1",
        "upstream_commit": driver.UPSTREAM_COMMIT,
        "tools": [
            {
                "name": "echo",
                "description": "echo(text: string) -> string",
                "input_schema": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                    "additionalProperties": False,
                },
                "read_only": True,
                "network": False,
                "implementation": "replay_table",
                "cases": cases,
            }
        ],
    }


def build_request(
    *,
    release_id: str,
    tool_sha: str,
    model_sha: str,
    streaming: bool,
) -> dict[str, Any]:
    mode = "full" if release_id.endswith(".full") else "planner_component"
    identity = {
        "method_id": "llm_compiler",
        "release_id": release_id,
        "upstream_commit": driver.UPSTREAM_COMMIT,
        "state_hash": "a" * 64,
        "payload": {
            "task_id": "fixture-1",
            "query": "question",
            "instruction": "",
            "messages": [],
            "tool_specs": [
                {
                    "name": "echo",
                    "description": "echo(text: string) -> string",
                    "input_schema": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                        "additionalProperties": False,
                    },
                    "read_only": True,
                    "network": False,
                }
            ],
            "controls": {
                "max_tasks": 4,
                "max_replans": 1,
                "concurrency": 2,
                "timeout_seconds": 30.0,
                "streaming": streaming,
            },
            "prompt_hash": driver._canonical_hash(prompts()),
            "execution_mode": mode,
            "seed": 7,
        },
        "dependency_fingerprints": {
            "backend": {"replay_manifest_sha256": model_sha},
            "tools": {"tool_manifest_sha256": tool_sha},
        },
        "minimum_reservation": {"model_calls": 1},
        "protocol_version": driver.PROTOCOL_VERSION,
    }
    return {**identity, "request_id": driver._canonical_hash(identity)}


def prepare(
    tmp_path: Path,
    *,
    release_id: str,
    entries: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    streaming: bool,
) -> tuple[driver.ValidatedRequest, driver.ReplayBundle, driver.ToolReplay]:
    model_path = tmp_path / "model-replay.json"
    model_sha = write_json(
        model_path,
        {
            "schema_version": "agentinputlab.llmcompiler.model-replay.v1",
            "upstream_commit": driver.UPSTREAM_COMMIT,
            "seed": 7,
            "prompts": prompts(),
            "prompt_hash": driver._canonical_hash(prompts()),
            "responses": entries,
        },
    )
    tool_path = tmp_path / "tool-replay.json"
    tool_sha = write_json(tool_path, tool_manifest(cases=cases))
    request = driver.validate_request(
        build_request(
            release_id=release_id,
            tool_sha=tool_sha,
            model_sha=model_sha,
            streaming=streaming,
        )
    )
    return (
        request,
        driver.load_model_replay(model_path, request),
        driver.load_tool_replay(tool_path, request),
    )


def test_all_required_dependency_versions_and_import_origins_are_verified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = driver._verify_static_manifests()
    required = manifest["legacy_dependencies"]["required_versions"]
    site_packages = tmp_path / "site-packages"
    modules: dict[str, ModuleType] = {}
    distributions: dict[str, Any] = {}
    checked: list[str] = []

    class FakeDistribution:
        def __init__(self, name: str, version: str) -> None:
            self.name = name
            self.version = version

        def locate_file(self, relative: str) -> Path:
            return site_packages / relative

    for name, version in required.items():
        import_name = name.replace("-", "_")
        package = site_packages / import_name
        package.mkdir(parents=True)
        module = ModuleType(import_name)
        module.__file__ = str(package / "__init__.py")
        modules[import_name] = module
        distributions[name] = FakeDistribution(name, version)

    def distribution(name: str) -> Any:
        checked.append(name)
        return distributions[name]

    monkeypatch.setattr(driver.importlib.metadata, "distribution", distribution)
    monkeypatch.setattr(driver.importlib, "import_module", modules.__getitem__)

    driver._verify_required_dependencies(manifest)
    assert checked == list(required)

    modules["openai"].__file__ = str(tmp_path / "shadow" / "openai" / "__init__.py")
    with pytest.raises(driver.WorkerFailure) as captured:
        driver._verify_required_dependencies(manifest)
    assert captured.value.code == "LEGACY_DEPENDENCY_ORIGIN_MISMATCH"


def test_real_checkout_matches_commit_license_and_source_hashes() -> None:
    if not _is_git_checkout(REAL_CHECKOUT):
        pytest.skip("pinned LLMCompiler checkout is not available")
    manifest = driver.verify_checkout(REAL_CHECKOUT)
    assert manifest["upstream"]["license_spdx"] == "MIT"
    assert manifest["upstream"]["commit"] == driver.UPSTREAM_COMMIT
    assert len(manifest["source_files"]) >= 10


def test_real_checkout_import_smoke_fails_clearly_without_legacy_langchain() -> None:
    if not _is_git_checkout(REAL_CHECKOUT):
        pytest.skip("pinned LLMCompiler checkout is not available")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "workers/llmcompiler/driver.py"),
            "--checkout",
            str(REAL_CHECKOUT),
            "--smoke-import",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    lines = result.stdout.splitlines()
    assert len(lines) == 1
    response = json.loads(lines[0])
    if response["status"] == "ok":
        assert result.returncode == 0
        assert response["smoke_import"] is True
    else:
        assert result.returncode == 2
        assert response["failure_code"] in {
            "LEGACY_DEPENDENCY_MISSING",
            "LEGACY_DEPENDENCY_VERSION_MISMATCH",
        }
        assert "langchain" in response["message"]
    assert "LLMCompiler planner response" not in result.stdout


def test_planner_component_calls_upstream_streaming_planner(tmp_path: Path) -> None:
    request, model, tools = prepare(
        tmp_path,
        release_id="plan.llm_compiler.planner_component",
        entries=[response_entry("planner", "planner:question:False", "PLAN:first")],
        cases=[],
        streaming=True,
    )
    upstream, calls = fake_upstream()

    response = asyncio.run(driver.execute_request(request, upstream, model, tools))

    assert response["payload"]["termination"] == "planner_complete"
    assert response["payload"]["scheduler_events"] == []
    assert response["payload"]["tasks"] == [
        {
            "task_id": 1,
            "tool_name": "echo",
            "arguments": {"text": "first"},
            "dependencies": [],
        }
    ]
    assert response["usage"]["model_calls"] == 1
    assert response["usage"]["tool_calls"] == 0
    assert calls["aplan"] == 1
    assert calls["task_fetcher"] == 0


def test_timeout_covers_streaming_planner_and_leaves_no_task(tmp_path: Path) -> None:
    request, model, tools = prepare(
        tmp_path,
        release_id="plan.llm_compiler.planner_component",
        entries=[response_entry("planner", "planner:question:False", "PLAN:first")],
        cases=[],
        streaming=True,
    )
    request = replace(
        request,
        controls=replace(request.controls, timeout_seconds=0.01),
    )
    upstream, _calls = fake_upstream()
    planner_type = upstream.planner.Planner

    class HangingPlanner(planner_type):
        async def aplan(self, **_: Any) -> None:
            await asyncio.Event().wait()

    upstream.planner.Planner = HangingPlanner

    async def scenario() -> driver.WorkerFailure:
        existing = set(asyncio.all_tasks())
        with pytest.raises(driver.WorkerFailure) as captured:
            await driver.execute_request(request, upstream, model, tools)
        await asyncio.sleep(0)
        assert set(asyncio.all_tasks()) <= existing
        return captured.value

    failure = asyncio.run(scenario())
    assert failure.code == "RUN_TIMEOUT"
    assert failure.consumed is False
    assert failure.usage is not None and failure.usage["model_calls"] == 0


def test_streaming_planner_failure_sends_sentinel_and_fails_closed(tmp_path: Path) -> None:
    request, model, tools = prepare(
        tmp_path,
        release_id="plan.llm_compiler.planner_component",
        entries=[response_entry("planner", "planner:question:False", "PLAN:first")],
        cases=[],
        streaming=True,
    )
    upstream, _calls = fake_upstream()
    planner_type = upstream.planner.Planner

    class FailingPlanner(planner_type):
        async def aplan(self, **_: Any) -> None:
            raise RuntimeError("private planner failure")

    upstream.planner.Planner = FailingPlanner

    async def scenario() -> driver.WorkerFailure:
        existing = set(asyncio.all_tasks())
        with pytest.raises(driver.WorkerFailure) as captured:
            await asyncio.wait_for(
                driver.execute_request(request, upstream, model, tools),
                timeout=0.5,
            )
        await asyncio.sleep(0)
        assert set(asyncio.all_tasks()) <= existing
        return captured.value

    failure = asyncio.run(scenario())
    assert failure.code == "PLANNER_STREAM_FAILURE"
    assert "private planner failure" not in failure.public_message


def test_planner_component_bridge_e2e_uses_upstream_planner(tmp_path: Path) -> None:
    model_path = tmp_path / "model-replay.json"
    model_sha = write_json(
        model_path,
        {
            "schema_version": "agentinputlab.llmcompiler.model-replay.v1",
            "upstream_commit": driver.UPSTREAM_COMMIT,
            "seed": 7,
            "prompts": prompts(),
            "prompt_hash": driver._canonical_hash(prompts()),
            "responses": [response_entry("planner", "planner:question:False", "PLAN:first")],
        },
    )
    tool_path = tmp_path / "tool-replay.json"
    tool_sha = write_json(tool_path, tool_manifest(cases=[]))
    upstream, calls = fake_upstream()
    worker = InProcessReplayWorker(
        upstream=upstream,
        model_path=model_path,
        tool_path=tool_path,
    )
    context = InterventionContext(
        backend=FingerprintedBackend(model_sha),
        tools=ManifestToolConnector(tool_sha),
        external_runtime=worker,
        seed=7,
    )

    patch = LLMCompilerPlannerBridge(
        prompt_hash=driver._canonical_hash(prompts()),
        controls=LLMCompilerControls(
            max_tasks=4,
            max_replans=1,
            concurrency=2,
            timeout_seconds=30.0,
            streaming=True,
        ),
    ).propose(
        FrozenAgentState(query="question", task_id="fixture-1"),
        context,
    )

    assert worker.calls == 1
    assert calls["aplan"] == 1
    assert calls["task_fetcher"] == 0
    assert len(patch.metadata["task_dag"]) == 1
    assert patch.metadata["task_dag"][0]["tool_name"] == "echo"
    assert patch.metadata["task_dag"][0]["arguments"]["text"] == "first"
    assert patch.trace.counters["model_calls"] == 1


def test_full_worker_runs_upstream_scheduler_join_and_replan(tmp_path: Path) -> None:
    request, model, tools = prepare(
        tmp_path,
        release_id="plan.llm_compiler.full",
        entries=[
            response_entry("planner", "planner:question:False", "PLAN:first"),
            response_entry(
                "join",
                "join:question:FIRST:False",
                "Thought: need more\nAction: Replan(more)",
            ),
            response_entry("planner", "planner:question:True", "PLAN:second"),
            response_entry(
                "join",
                "join:question:SECOND:True",
                "Thought: done\nAction: Finish(done)",
            ),
        ],
        cases=[
            {"arguments": ["first"], "output": "FIRST"},
            {"arguments": ["second"], "output": "SECOND"},
        ],
        streaming=True,
    )
    upstream, calls = fake_upstream()

    response = asyncio.run(driver.execute_request(request, upstream, model, tools))

    payload = response["payload"]
    assert payload["termination"] == "finish"
    assert payload["final_answer"] == "done"
    assert payload["replans"] == 1
    assert [task["task_id"] for task in payload["tasks"]] == [1, 2]
    assert [event["state"] for event in payload["scheduler_events"]] == [
        "planned",
        "ready",
        "running",
        "completed",
        "planned",
        "ready",
        "running",
        "completed",
    ]
    assert response["usage"] == {
        "model_calls": 4,
        "retrieval_calls": 0,
        "tool_calls": 2,
        "input_tokens": 20,
        "output_tokens": 12,
    }
    assert calls == {
        "planner": 0,
        "aplan": 2,
        "task_fetcher": 2,
        "join": 2,
        "compiler": 1,
    }


@pytest.mark.parametrize("streaming", [False, True])
def test_replan_reuses_completed_dependencies_without_duplicate_tools_or_scratchpad(
    tmp_path: Path, streaming: bool,
) -> None:
    request, model, tools = prepare(
        tmp_path,
        release_id="plan.llm_compiler.full",
        entries=[
            response_entry("planner", "planner:question:False", "PLAN:first"),
            response_entry("join", "join:question:FIRST:False", "Action: Replan(more)"),
            response_entry("planner", "planner:question:True", "PLAN:second"),
            response_entry("join", "join:question:SECOND:True", "Action: Finish(done)"),
        ],
        cases=[{"arguments": ["first"], "output": "FIRST"},
               {"arguments": ["FIRST"], "output": "SECOND"}],
        streaming=streaming,
    )
    upstream, calls = fake_upstream()
    original_planner = upstream.planner.Planner
    original_tfu = upstream.task_fetching.TaskFetchingUnit

    class NativeStylePlanner(original_planner):
        def _tasks(self, response: str) -> dict[int, FakeTask]:
            task_id = 1 if response == "PLAN:first" else 3
            return {
                task_id: FakeTask(task_id, "echo", self.tools[0].func,
                    ("first",) if task_id == 1 else ("$1",),
                    () if task_id == 1 else (1,)),
                task_id + 1: FakeTask(task_id + 1, "join", lambda: None, (),
                    tuple(range(1, task_id + 1)), is_join=True),
            }

    class NativeStyleTFU(original_tfu):
        async def _run_task(self, task: FakeTask) -> None:
            if task.args == ("$1",):
                task.args = (self.tasks[1].observation,)
            await super()._run_task(task)

    upstream.planner.Planner = NativeStylePlanner
    upstream.task_fetching.TaskFetchingUnit = NativeStyleTFU
    upstream.compiler.TaskFetchingUnit = NativeStyleTFU
    response = asyncio.run(driver.execute_request(request, upstream, model, tools))
    assert response["payload"]["final_answer"] == "done"
    assert response["payload"]["replans"] == 1
    assert [task["task_id"] for task in response["payload"]["tasks"]] == [1, 3]
    assert response["payload"]["tasks"][1]["dependencies"] == [1]
    assert response["payload"]["tasks"][1]["arguments"] == {"text": "FIRST"}
    assert response["usage"]["model_calls"] == 4
    assert response["usage"]["tool_calls"] == 2
    assert calls["task_fetcher"] == 2
    assert len(response["payload"]["scheduler_events"]) == 8
    assert upstream.compiler.TaskFetchingUnit is NativeStyleTFU


def test_replan_dependency_cache_is_request_local_and_does_not_iterate_history() -> None:
    async def scenario() -> None:
        recorder = driver.TraceRecorder(frozenset({"echo"}), 4, 2)
        tfu = driver._traced_task_fetching_unit(FakeTaskFetchingUnit, recorder)

        async def echo(text: str) -> str:
            return text.upper()

        first = tfu()
        first.set_tasks({1: FakeTask(1, "echo", echo, ("first",)),
                         2: FakeTask(2, "join", lambda: None, (), (1,), is_join=True)})
        await first.schedule()
        second = tfu()
        assert second.tasks[1].observation == "FIRST"
        assert second.tasks_done[1].is_set() and second.tasks_done[2].is_set()
        assert list(second.tasks.values()) == []
        assert list(second.tasks_done.values()) == []
        assert not second.remaining_tasks
        other = driver._traced_task_fetching_unit(FakeTaskFetchingUnit,
            driver.TraceRecorder(frozenset({"echo"}), 4, 2))()
        with pytest.raises(KeyError):
            other.tasks[1]
        with pytest.raises(KeyError):
            other.tasks_done[1]

    asyncio.run(scenario())


def test_full_worker_rejects_unknown_dependency_before_tool_or_join(tmp_path: Path) -> None:
    request, model, tools = prepare(
        tmp_path,
        release_id="plan.llm_compiler.full",
        entries=[response_entry("planner", "planner:question:False", "PLAN:first")],
        cases=[{"arguments": ["first"], "output": "FIRST"}],
        streaming=True,
    )
    upstream, calls = fake_upstream()
    planner_type = upstream.planner.Planner

    class UnknownDependencyPlanner(planner_type):
        def _tasks(self, response: str) -> dict[int, FakeTask]:
            del response
            return {
                1: FakeTask(1, "echo", self.tools[0].func, ("first",), (999,)),
                101: FakeTask(101, "join", lambda: None, (), (1,), is_join=True),
            }

    upstream.planner.Planner = UnknownDependencyPlanner

    async def scenario() -> driver.WorkerFailure:
        with pytest.raises(driver.WorkerFailure) as captured:
            await asyncio.wait_for(
                driver.execute_request(request, upstream, model, tools),
                timeout=0.5,
            )
        return captured.value

    failure = asyncio.run(scenario())
    assert failure.code == "UPSTREAM_PLAN_INVALID"
    assert tools.calls == 0
    assert calls["join"] == 0


def test_full_worker_streaming_planner_failure_cannot_deadlock_scheduler(
    tmp_path: Path,
) -> None:
    request, model, tools = prepare(
        tmp_path,
        release_id="plan.llm_compiler.full",
        entries=[response_entry("planner", "planner:question:False", "PLAN:first")],
        cases=[],
        streaming=True,
    )
    upstream, calls = fake_upstream()
    planner_type = upstream.planner.Planner

    class FailingPlanner(planner_type):
        async def aplan(self, **_: Any) -> None:
            raise RuntimeError("private planner failure")

    upstream.planner.Planner = FailingPlanner

    async def scenario() -> driver.WorkerFailure:
        existing = set(asyncio.all_tasks())
        with pytest.raises(driver.WorkerFailure) as captured:
            await asyncio.wait_for(
                driver.execute_request(request, upstream, model, tools),
                timeout=0.5,
            )
        await asyncio.sleep(0)
        assert set(asyncio.all_tasks()) <= existing
        return captured.value

    failure = asyncio.run(scenario())
    assert failure.code == "PLANNER_STREAM_FAILURE"
    assert tools.calls == 0
    assert calls["join"] == 0


def test_full_bridge_e2e_is_controlled_without_official_attestation(tmp_path: Path) -> None:
    model_path = tmp_path / "model-replay.json"
    model_sha = write_json(
        model_path,
        {
            "schema_version": "agentinputlab.llmcompiler.model-replay.v1",
            "upstream_commit": driver.UPSTREAM_COMMIT,
            "seed": 7,
            "prompts": prompts(),
            "prompt_hash": driver._canonical_hash(prompts()),
            "responses": [
                response_entry("planner", "planner:question:False", "PLAN:first"),
                response_entry(
                    "join",
                    "join:question:FIRST:False",
                    "Thought: done\nAction: Finish(done)",
                ),
            ],
        },
    )
    tool_path = tmp_path / "tool-replay.json"
    tool_sha = write_json(
        tool_path,
        tool_manifest(cases=[{"arguments": ["first"], "output": "FIRST"}]),
    )
    upstream, _calls = fake_upstream()
    worker = InProcessReplayWorker(
        upstream=upstream,
        model_path=model_path,
        tool_path=tool_path,
    )
    context = InterventionContext(
        backend=FingerprintedBackend(model_sha),
        tools=ManifestToolConnector(tool_sha),
        external_runtime=worker,
        seed=7,
    )

    result = LLMCompilerFullBridge(
        prompt_hash=driver._canonical_hash(prompts()),
        controls=LLMCompilerControls(max_tasks=4, max_replans=1, concurrency=2),
    ).run(FrozenAgentState(query="question", task_id="fixture-1"), context)

    assert result.final_answer == "done"
    assert result.profile_kind == "controlled_adapter"
    assert result.metadata["implementation_origin"] == "controlled_external_adapter"
    assert result.metadata["profile_attestation"] == "controlled_default"


def test_failure_usage_keeps_consumed_tool_calls(tmp_path: Path) -> None:
    request, model, tools = prepare(
        tmp_path,
        release_id="plan.llm_compiler.full",
        entries=[
            response_entry("planner", "planner:question:False", "PLAN:first"),
            response_entry(
                "join",
                "join:question:FIRST:False",
                "Thought: done\nAction: Finish(done)",
            ),
            response_entry("planner", "unused prompt", "PLAN:unused"),
        ],
        cases=[{"arguments": ["first"], "output": "FIRST"}],
        streaming=True,
    )
    upstream, _calls = fake_upstream()

    with pytest.raises(driver.WorkerFailure) as captured:
        asyncio.run(driver.execute_request(request, upstream, model, tools))

    assert captured.value.code == "MODEL_REPLAY_UNUSED"
    assert captured.value.consumed is True
    assert captured.value.usage == {
        "model_calls": 2,
        "retrieval_calls": 0,
        "tool_calls": 1,
        "input_tokens": 10,
        "output_tokens": 6,
    }


def test_tool_replay_miss_aborts_before_join(tmp_path: Path) -> None:
    request, model, tools = prepare(
        tmp_path,
        release_id="plan.llm_compiler.full",
        entries=[response_entry("planner", "planner:question:False", "PLAN:first")],
        cases=[],
        streaming=True,
    )
    upstream, calls = fake_upstream()

    with pytest.raises(driver.WorkerFailure) as captured:
        asyncio.run(driver.execute_request(request, upstream, model, tools))

    assert captured.value.code == "TOOL_REPLAY_MISS"
    assert captured.value.consumed is True
    assert captured.value.usage == {
        "model_calls": 1,
        "retrieval_calls": 0,
        "tool_calls": 1,
        "input_tokens": 5,
        "output_tokens": 3,
    }
    assert calls["join"] == 0


def test_network_or_side_effect_tool_is_rejected_before_runtime() -> None:
    request = build_request(
        release_id="plan.llm_compiler.full",
        tool_sha="b" * 64,
        model_sha="c" * 64,
        streaming=True,
    )
    request["payload"]["tool_specs"][0]["network"] = True
    identity = {key: value for key, value in request.items() if key != "request_id"}
    request["request_id"] = driver._canonical_hash(identity)

    with pytest.raises(driver.WorkerFailure, match="read-only, no-network") as captured:
        driver.validate_request(request)

    assert captured.value.code == "TOOL_POLICY_DENIED"
    assert captured.value.consumed is False


def test_request_cannot_select_an_import_command_or_network_target() -> None:
    request = build_request(
        release_id="plan.llm_compiler.full",
        tool_sha="b" * 64,
        model_sha="c" * 64,
        streaming=True,
    )
    request["payload"]["module"] = "attacker.module"
    identity = {key: value for key, value in request.items() if key != "request_id"}
    request["request_id"] = driver._canonical_hash(identity)

    with pytest.raises(driver.WorkerFailure, match="unsupported or missing fields"):
        driver.validate_request(request)


def test_model_prompt_hash_mismatch_fails_before_consumption(tmp_path: Path) -> None:
    _request, model, _ = prepare(
        tmp_path,
        release_id="plan.llm_compiler.planner_component",
        entries=[response_entry("planner", "different prompt", "PLAN:first")],
        cases=[],
        streaming=True,
    )
    replay = driver._ReplayModel(model.entries)

    with pytest.raises(driver.WorkerFailure) as captured:
        asyncio.run(replay.apredict("planner:question:False"))

    assert captured.value.code == "MODEL_PROMPT_MISMATCH"
    assert captured.value.consumed is False
    assert replay.usage()["model_calls"] == 0


def test_patch_manifest_is_separate_and_hash_bound() -> None:
    manifest = driver._verify_static_manifests()
    patch = manifest["patch_manifest"]
    patch_path = ROOT / "workers/llmcompiler" / patch["path"]
    assert driver._sha256_bytes(patch_path.read_bytes()) == patch["sha256"]
    patch_value = json.loads(patch_path.read_text(encoding="utf-8"))
    assert len(patch_value["patches"]) >= 5
    assert all(item["algorithm_logic_changed"] is False for item in patch_value["patches"])
