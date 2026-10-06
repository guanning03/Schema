from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

from env import ArcEnv

from .agent import WorldModelAgent
from .events import FanoutSink
from .obs import ConsoleSink, JsonlSink

_TMP = Path(__file__).resolve().parents[2] / "tmp"

_DEFAULT_MODEL = {"claude": "claude-opus-4-8", "codex-cli": "gpt-5.5"}
_POOL_BASE = {"claude": ".claude-arc-agent", "codex-cli": ".codex-arc-agent"}


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "task"


def _max_event_seq(path: Path) -> int:
    if not path.is_file():
        return 0
    mx = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            mx = max(mx, int(json.loads(line).get("seq", 0)))
        except Exception:
            continue
    return mx


def _pool_dirs(spec: str, base: str) -> list[str]:
    home = Path.home()

    def _d(n: str) -> str:
        if n == "1":
            plain = home / base
            return str(plain if plain.is_dir() else home / f"{base}-1")
        return str(home / f"{base}-{n}")

    parts = re.split(r"[,:\s]+", spec.strip()) if re.search(r"[,:\s]", spec) else list(spec.strip())
    dirs = [_d(n) for n in parts if n]
    return [d for d in dirs if Path(d).is_dir()]


def _resume_game(workdir: Path) -> "str | None":
    r = subprocess.run(["git", "-C", str(workdir), "show", "HEAD:run.json"],
                       capture_output=True, text=True)
    txt = r.stdout if r.returncode == 0 and r.stdout.strip() else (
        (workdir / "run.json").read_text(encoding="utf-8") if (workdir / "run.json").is_file() else None)
    if not txt:
        return None
    try:
        gid = (json.loads(txt) or {}).get("game_id")
    except Exception:
        return None
    return gid.split("-")[0] if gid else None


def main() -> int:
    p = argparse.ArgumentParser(prog="python -m agent.world_model.solve")
    p.add_argument("--game", default="ls20")
    p.add_argument("--provider", choices=["claude", "codex-cli"], default="claude")
    p.add_argument("--model", default=None)
    p.add_argument("--effort", default="max")
    p.add_argument("--reasoning", default=None)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--pool", default=None, metavar="SEQ")
    p.add_argument("--jail", action="store_true")
    p.add_argument("--jail-runtime", choices=["enroot", "podman", "docker"], default="podman")
    p.add_argument("--workdir", type=Path, default=None)
    p.add_argument("--resume", type=Path, default=None, metavar="WORKDIR")
    args = p.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    if args.jail and args.provider != "claude":
        p.error("--jail supports --provider claude only")

    if args.pool:
        dirs = _pool_dirs(args.pool, _POOL_BASE[args.provider])
        if not dirs:
            p.error(f"--pool: no existing account dirs resolved from {args.pool!r}")
        if args.provider == "codex-cli":
            os.environ["ARC_CODEX_POOL"] = ":".join(dirs)
        else:
            os.environ["ARC_CLAUDE_POOL"] = ":".join(dirs)
            os.environ["CLAUDE_CONFIG_DIR"] = dirs[0]
        print(f"account pool (priority): {', '.join(Path(d).name for d in dirs)}")

    game = args.game
    if args.resume is not None:
        g = _resume_game(Path(args.resume).expanduser())
        if g:
            if g != args.game:
                print(f"resume: using game '{g}' from the workdir (ignoring --game {args.game!r})")
            game = g

    env = ArcEnv(game)
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    resume = args.resume is not None
    if resume:
        workdir = args.resume.expanduser().resolve()
        if not workdir.is_dir():
            p.error(f"--resume: no such workdir: {workdir}")
    elif args.workdir:
        workdir = Path(args.workdir)
    else:
        base = _TMP / f"{_safe_name(env.game_id)}_{start_time}"
        workdir = base
        n = 1
        while workdir.exists():
            n += 1
            workdir = Path(f"{base}_{n}")
    workdir.mkdir(parents=True, exist_ok=True)

    start_seq = _max_event_seq(workdir / "events.jsonl") if resume else 0
    events = FanoutSink([ConsoleSink(), JsonlSink(workdir / "events.jsonl")], start_seq=start_seq)

    orch = None
    claude_container = None
    executor = None
    if args.jail:
        from sandbox.orchestrator import SandboxOrchestrator

        orch = SandboxOrchestrator(workdir=workdir, runtime=args.jail_runtime)
        claude_container, executor = orch.start()

    agent = WorldModelAgent(
        arc_env=env.wrapper,
        game_id=env.game_id,
        provider=args.provider,
        model=args.model or _DEFAULT_MODEL[args.provider],
        effort=args.effort,
        reasoning=args.reasoning,
        workdir=workdir,
        events=events,
        resume=resume,
        executor=executor,
        claude_container=claude_container,
    )
    agent.MAX_ACTIONS = args.steps

    print(f"world_model solve {env.game_id} with {args.provider}; max steps={args.steps}")
    if resume:
        print(
            f"resumed from {workdir}: {agent.resumed_transitions} transition(s), "
            f"world model {'loaded' if agent.world is not None else ('text-only' if agent.code else 'none')}"
        )
    print(f"workdir: {workdir}")
    try:
        agent.main()
    finally:
        if orch is not None:
            orch.stop()
    print(f"artifacts: {workdir / 'world_model.py'}, {workdir / 'notes.md'}, {workdir / 'events.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
