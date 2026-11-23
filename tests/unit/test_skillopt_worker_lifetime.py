"""Real process-lifetime tests; the native episode alone is a no-model fixture."""

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

SOURCE = Path(
    next(iter(importlib.util.find_spec("agentinputlab").submodule_search_locations))
).parent
WORKERS = Path(next(iter(importlib.util.find_spec("workers").submodule_search_locations))).parent


def wait_for(predicate, timeout=15):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def alive(pid):
    result = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, check=False
    )
    return bool(result.stdout.strip()) and not result.stdout.strip().startswith("Z")


@pytest.mark.skipif(
    os.name != "posix", reason="isolated native worker uses POSIX process semantics"
)
@pytest.mark.parametrize("mode", ["complete", "parent_exit", "episode_timeout", "startup_timeout"])
def test_actual_launcher_worker_lifetime(tmp_path, mode):
    worker = tmp_path / "fixture-python"
    worker.write_text(
        f"#!{sys.executable}\n"
        "import sys,json,os,time;from pathlib import Path\n"
        f"Path({str(tmp_path / 'worker-started.json')!r}).write_text(json.dumps({{'pid':os.getpid()}}))\n"
        f"time.sleep({12 if mode == 'startup_timeout' else 0})\n"
        f"sys.path.insert(0, {str(WORKERS)!r})\n"
        f"sys.path.insert(0, {str(SOURCE)!r})\n"
        "from agentinputlab.recipes import skillopt_toolsandbox as native\n"
        "def episode_fixture(path):\n"
        "    job=json.loads(path.read_text())\n"
        "    out=native.Path(job['output'])\n"
        "    (out/'ready.json').write_text(json.dumps({'pid':os.getpid()}))\n"
        f"    time.sleep({0 if mode == 'complete' else 60})\n"
        "    native._save(out/'episode-terminal.json', {'status':'scored'})\n"
        "    return 0\n"
        "native._worker=episode_fixture\n"
        "assert sys.argv[1]=='-c'\n"
        "code=sys.argv[2];sys.argv=['-c']+sys.argv[3:]\n"
        "exec(code)\n"
    )
    worker.chmod(0o700)
    output = tmp_path / "episode"
    output.mkdir()
    configuration = dict(
        worker_python=str(worker), episode_timeout_seconds=8 if mode.endswith("timeout") else 90
    )
    script = (
        "import sys;from pathlib import Path;"
        f"sys.path.insert(0,{str(WORKERS)!r});"
        f"sys.path.insert(0,{str(SOURCE)!r});"
        "from agentinputlab.recipes.skillopt_toolsandbox import _launch;"
        f"_launch({configuration!r},"
        f"'fixture','skill',Path({str(output)!r}))"
    )
    pid = None
    with (tmp_path / "parent.log").open("w") as log:
        parent = subprocess.Popen([sys.executable, "-c", script], stdout=log, stderr=log)
        try:
            # The configured timeout includes imports/source verification, not
            # just the episode. On cold NFS it may correctly expire before
            # episode_fixture starts; still assert the actual worker exits.
            receipt = (tmp_path / "worker-started.json" if mode.endswith("timeout")
                       else output / "ready.json")
            assert wait_for(receipt.exists), (
                tmp_path / "parent.log"
            ).read_text() + (output / "worker.log").read_text()
            pid = json.loads(receipt.read_text())["pid"]
            if mode == "parent_exit":
                parent.terminate()
                parent.wait(timeout=5)
                assert wait_for(lambda: not alive(pid), timeout=5), (
                    "native episode outlived its trainer"
                )
                assert not (output / "episode-terminal.json").exists()
            elif mode == "complete":
                assert parent.wait(timeout=15) == 0, (tmp_path / "parent.log").read_text()
                assert (
                    json.loads((output / "episode-terminal.json").read_text())["status"] == "scored"
                )
                assert json.loads((output / "worker-exit.json").read_text())["returncode"] == 0
            else:
                assert parent.wait(timeout=20) != 0
                assert not alive(pid)
                terminal = json.loads((output / "worker-exit.json").read_text())
                assert terminal["execution_timeout"] and not terminal["automatic_retry"]
        finally:
            if parent.poll() is None:
                parent.terminate()
                parent.wait(timeout=5)
            if pid is not None and alive(pid):
                os.kill(pid, signal.SIGTERM)
                assert wait_for(lambda: not alive(pid), timeout=5)


@pytest.mark.skipif(os.name != "posix", reason="POSIX lifetime pipe")
def test_parent_already_exited_prevents_episode_start():
    read_fd, write_fd = os.pipe()
    os.close(write_fd)
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                f"import sys;sys.path.insert(0,{str(WORKERS)!r});sys.path.insert(0,{str(SOURCE)!r});"
                "from agentinputlab.recipes.skillopt_toolsandbox import _watch_parent;"
                f"_watch_parent({read_fd});print('EPISODE_STARTED')",
            ],
            pass_fds=(read_fd,),
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    finally:
        os.close(read_fd)
    assert result.returncode != 0
    assert "trainer exited before worker startup" in result.stderr
    assert "EPISODE_STARTED" not in result.stdout
