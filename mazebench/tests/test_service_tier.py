"""MazeBench asks Codex for the Ultrafast service tier by default; `--service-tier standard` turns it off."""
from __future__ import annotations

from agent.world_model.codex_cli_driver import CodexCliDriver
from agent.world_model.solve import build_parser


def _driver(tmp_path, monkeypatch, tier):
    monkeypatch.delenv("ARC_CODEX_POOL", raising=False)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    d = CodexCliDriver(model="gpt-6-astra", reasoning="high", cwd=str(tmp_path), service_tier=tier)
    d._exe = "codex"
    return d


def test_default_tier_is_ultrafast():
    assert build_parser().parse_args([]).service_tier == "ultrafast"


def test_tier_is_passed_to_codex(tmp_path, monkeypatch):
    cmd = _driver(tmp_path, monkeypatch, "ultrafast")._build_cmd(None)
    assert cmd[cmd.index("service_tier=ultrafast") - 1] == "-c"


def test_no_tier_means_standard(tmp_path, monkeypatch):
    cmd = _driver(tmp_path, monkeypatch, None)._build_cmd(None)
    assert not any(str(c).startswith("service_tier=") for c in cmd)
