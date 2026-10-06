from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

from env.maze_env import MazeEnv

from .agent import ResumeError, WorldModelAgent
from .events import FanoutSink
from .obs.jsonl import JsonlSink
from .tools import exec_sandbox_binary

_TMP = Path("tmp")

ROOM = "level_HxI"
VIEW = "top-diagonal"
YAW = 0
HIDE_NAMES_SEED = "1"

_DEFAULT_MODEL = {"codex-cli": "gpt-6-astra", "claude": None}
_POOL_BASE = {"codex-cli": ".codex-arc-agent", "claude": ".claude-arc-agent"}


def _pool_dirs(spec: str, base: str) -> list:
    home = Path.home()
    parts = re.split(r"[,:\s]+", spec.strip()) if re.search(r"[,:\s]", spec) else list(spec.strip())
    out = []
    for n in parts:
        if not n:
            continue
        d = home / base if n == "1" and (home / base).is_dir() else home / f"{base}-{n}"
        if d.is_dir():
            out.append(str(d))
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agent.world_model.solve")
    p.add_argument("--provider", default="codex-cli", choices=["codex-cli", "claude"])
    p.add_argument("--model", default=None)
    p.add_argument("--reasoning", default="high")
    p.add_argument("--service-tier", default="ultrafast",
                   help="Codex service tier: ultrafast (default), priority (Fast), or standard")
    p.add_argument("--effort", default=None)
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--max-hours", type=float, default=None)
    p.add_argument("--workdir", type=Path, default=None)
    p.add_argument("--resume", type=Path, default=None, metavar="WORKDIR")
    p.add_argument("--pool", default=None, metavar="SEQ")
    return p


def main(argv: "list | None" = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    try:
        exec_sandbox_binary()
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 2

    if args.pool:
        dirs = _pool_dirs(args.pool, _POOL_BASE[args.provider])
        if not dirs:
            print(f"--pool: no account dirs resolved from {args.pool!r}", file=sys.stderr)
            return 2
        if args.provider == "codex-cli":
            os.environ["ARC_CODEX_POOL"] = ":".join(dirs)
        else:
            os.environ["ARC_CLAUDE_POOL"] = ":".join(dirs)
            os.environ["CLAUDE_CONFIG_DIR"] = dirs[0]
        print("account pool (priority): " + ", ".join(Path(d).name for d in dirs))

    model = args.model or _DEFAULT_MODEL[args.provider]
    if args.resume:
        args.workdir = args.resume
    if args.workdir:
        workdir = Path(args.workdir)
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        base = _TMP / f"maze_{ROOM}_{stamp}"
        workdir, n = base, 1
        while workdir.exists():
            n += 1
            workdir = Path(f"{base}_{n}")
    workdir.mkdir(parents=True, exist_ok=True)

    env = MazeEnv(room=ROOM, view=VIEW, yaw=YAW, hide_names_seed=HIDE_NAMES_SEED)
    events = FanoutSink([JsonlSink(workdir / "events.jsonl")])
    tier = None if args.service_tier in ("", "standard", "default", "none") else args.service_tier

    agent = WorldModelAgent(
        env=env, provider=args.provider, model=model, reasoning=args.reasoning,
        effort=args.effort, workdir=workdir, events=events, max_hours=args.max_hours,
        max_actions=args.steps, resume=bool(args.resume), service_tier=tier)

    print(f"MazeBench room={ROOM} view={VIEW} yaw={YAW} | {args.provider}"
          f"/{model or 'default'} | max steps={args.steps}")
    print(f"workdir: {workdir}")
    if not args.resume:
        (workdir / "run.json").write_text(json.dumps({
            "engine": str(env.engine), "room": ROOM, "view": VIEW, "yaw": YAW,
            "hide_names": True, "hide_names_seed": HIDE_NAMES_SEED,
            "provider": args.provider, "model": model, "effort": args.effort,
            "reasoning": args.reasoning, "service_tier": tier or "standard",
            "max_actions": args.steps,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, indent=2), encoding="utf-8")

    try:
        agent.main()
    except ResumeError as e:
        print(f"\n{e}", file=sys.stderr)
        agent.cleanup()
        events.close()
        return 3
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
    finally:
        card = {}
        try:
            card = env.scorecard()
        except Exception:
            pass
        agent.cleanup()
        events.close()
        if card:
            res = card.get("result", {})
            rooms = card.get("rooms", {})
            print(f"─ done · gems {card.get('gems', {}).get('collected', 0)}/{card.get('gems', {}).get('total', '?')} "
                  f"({res.get('percent', 0):.0f}%) · rooms {rooms.get('visited', 0)}"
                  f"/{rooms.get('total', 0)} · actions {card.get('actions', {}).get('total', 0)}")
            (workdir / "scorecard.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    print(f"artifacts: {workdir/'world_model.py'}, {workdir/'notes.md'}, {workdir/'events.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
