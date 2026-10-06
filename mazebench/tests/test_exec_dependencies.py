import pytest

from agent.world_model import solve, tools


def test_missing_bubblewrap_stops_before_creating_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(tools.sys, "platform", "linux")
    monkeypatch.setattr(tools.shutil, "which", lambda name: None)

    def unexpected_engine(**kwargs):
        pytest.fail("the engine must not start before the dependency check")

    monkeypatch.setattr(solve, "MazeEnv", unexpected_engine)
    workdir = tmp_path / "run"
    assert solve.main(["--workdir", str(workdir)]) == 2
    assert not workdir.exists()
    assert "apt-get install bubblewrap" in capsys.readouterr().err


def test_unsupported_platform_has_actionable_error(monkeypatch):
    monkeypatch.setattr(tools.sys, "platform", "darwin")
    with pytest.raises(RuntimeError, match="require Linux and bubblewrap"):
        tools.exec_sandbox_binary()


def test_installed_bubblewrap_is_accepted(monkeypatch):
    monkeypatch.setattr(tools.sys, "platform", "linux")
    monkeypatch.setattr(tools.shutil, "which", lambda name: "/usr/bin/bwrap")
    assert tools.exec_sandbox_binary() == "/usr/bin/bwrap"


def test_help_remains_available_without_bubblewrap(monkeypatch, capsys):
    monkeypatch.setattr(tools.shutil, "which", lambda name: None)
    with pytest.raises(SystemExit) as error:
        solve.main(["--help"])
    assert error.value.code == 0
    assert "--workdir" in capsys.readouterr().out
