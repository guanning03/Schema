from pathlib import Path
from types import SimpleNamespace

import pytest

from sandbox.agent_executor import Executor
from sandbox.claude_launch import ClaudeContainerLaunch


@pytest.mark.parametrize("source", ["default", "environment", "argument"])
def test_claude_login_directory_is_mounted(tmp_path, monkeypatch, source):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    config_dir = None
    expected = tmp_path / ".claude"
    if source in ("environment", "argument"):
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", "env-account")
        expected = tmp_path / "env-account"
    if source == "argument":
        config_dir = "pool-account"
        expected = tmp_path / "pool-account"

    runtime = SimpleNamespace(
        container_home="/tmp", proxy_in_container=True, needs_cwd_tmpfs=True,
        container_cmd=lambda **kwargs: kwargs,
    )
    launch = ClaudeContainerLaunch(
        container="claude", runtime=runtime, stage_src="/stage",
        proxy_sock_dir="/sock", workdir="/work", proxy_host="proxy", network="net",
    )
    command = launch.wrap(["claude", "--print"], config_dir, {})
    expected = str(expected.resolve())
    assert command["env"]["CLAUDE_CONFIG_DIR"] == expected
    assert (expected, expected, "rw") in command["mounts"]


@pytest.mark.parametrize("bad_code", ["def step(", "answer = 42"])
def test_rejected_model_preserves_previous_predictions(tmp_path, bad_code):
    executor = Executor(str(tmp_path), [])
    good_code = "def step(grid, action, x=None, y=None):\n    return [[grid[0][0] + action]]"
    assert executor.op_world_load({"code": good_code})["loaded"]
    assert not executor.op_world_load({"code": bad_code})["loaded"]
    assert executor.op_world_set_entry({"grid": [[1]], "level": 0}) == {"ok": True}
    result = executor.op_world_predict_step({
        "timeline": [], "entry": {"0": [[1]]}, "level": 0,
        "before_grid": [[1]], "action": 2,
    })
    assert result["grid"] == [[3]]


def test_successful_model_replaces_previous_model(tmp_path):
    executor = Executor(str(tmp_path), [])
    for value in (1, 2):
        assert executor.op_world_load({
            "code": f"def step(grid, action, x=None, y=None):\n    return [[{value}]]",
        })["loaded"]
        result = executor.op_world_predict_step({
            "timeline": [], "entry": {}, "level": 0,
            "before_grid": [[0]], "action": 1,
        })
        assert result["grid"] == [[value]]
