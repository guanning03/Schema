from __future__ import annotations

import argparse
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path

from env.dig_env import DigBenchAuthError, DigEnv, api_token, server_base_url

from .agent import WorldModelAgent
from .events import FanoutSink
from .obs.jsonl import JsonlSink

_TMP = Path(__file__).resolve().parents[2] / "tmp"


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


def _pool_dirs(spec: str, base: str = ".codex-arc-agent") -> list[str]:
    home = Path.home()

    def _d(n: str) -> str:
        if n == "1":
            plain = home / base
            return str(plain if plain.is_dir() else home / f"{base}-1")
        return str(home / f"{base}-{n}")

    parts = re.split(r"[,:\s]+", spec.strip()) if re.search(r"[,:\s]", spec) else list(spec.strip())
    dirs = [_d(n) for n in parts if n]
    return [d for d in dirs if Path(d).is_dir()]


def _read_run_json(workdir: Path) -> dict:
    p = workdir / "run.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def main() -> int:
    p = argparse.ArgumentParser(prog="python -m agent.world_model.solve")
    p.add_argument("--game", default="P-1")
    p.add_argument("--model", default="gpt-6-astra")
    p.add_argument("--reasoning", default="max")
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--max-hours", type=float, default=6.0)
    p.add_argument("--workdir", type=Path, default=None)
    p.add_argument("--resume", type=Path, default=None, metavar="WORKDIR")
    p.add_argument("--pool", default=None, metavar="SEQ")
    args = p.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    if args.pool:
        dirs = _pool_dirs(args.pool)
        if not dirs:
            p.error(f"--pool: no existing account dirs resolved from {args.pool!r}")
        os.environ["ARC_CODEX_POOL"] = ":".join(dirs)
        print(f"account pool (priority): {', '.join(Path(d).name for d in dirs)}")

    resume = args.resume is not None
    game = args.game
    session_id = None
    if resume:
        workdir = args.resume.expanduser().resolve()
        if not workdir.is_dir():
            p.error(f"--resume: no such workdir: {workdir}")
        meta = _read_run_json(workdir)
        if meta.get("game_id"):
            game = str(meta["game_id"])
        session_id = meta.get("session_id")
        if not session_id:
            p.error(f"--resume: no session_id in {workdir / 'run.json'}")
        print(f"resume: game {game}, session {session_id} from {workdir}")
    else:
        workdir = Path(args.workdir or (_TMP / f"{_safe_name(game)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    workdir.mkdir(parents=True, exist_ok=True)

    model_name = f"schema-codex-{args.model}"
    if not api_token():
        p.error("no DigBench API token: export DIGBENCH_API_TOKEN=... "
                "(mint one at https://digbench.ai/account/tokens)")
    notes = {"harness": "schema", "provider": "codex", "model": args.model, "reasoning": args.reasoning}
    try:
        env = DigEnv(game, model_name=model_name, model_version="0.1", notes=notes,
                     on_retry=lambda m: print(f"[bench] {m}", flush=True))
    except DigBenchAuthError as e:
        p.error(str(e))

    start_seq = _max_event_seq(workdir / "events.jsonl") if resume else 0
    events = FanoutSink([JsonlSink(workdir / "events.jsonl")], start_seq=start_seq)

    agent = WorldModelAgent(
        env=env,
        game_id=game,
        model=args.model,
        reasoning=args.reasoning,
        workdir=workdir,
        events=events,
        resume=resume,
        session_id=session_id,
        max_hours=args.max_hours,
        model_name=model_name,
    )
    agent.MAX_ACTIONS = args.steps

    print(f"schema solve {game} ({server_base_url()}) with codex/{args.model} "
          f"reasoning={args.reasoning}; max steps={args.steps}"
          + (f"; wall-clock cap {args.max_hours}h" if args.max_hours else ""))
    if resume:
        print(
            f"resumed from {workdir}: {agent.resumed_transitions} transition(s), "
            f"world model {'loaded' if agent.world is not None else ('text-only' if agent.code else 'none')}"
        )
    print(f"workdir: {workdir}")
    try:
        agent.main()
    finally:
        try:
            env.close()
        except Exception:
            pass
    last = agent.frames[-1] if agent.frames else None
    if last is not None:
        print(f"result: {last.state.status} — levels beaten {last.levels_beaten}"
              f"/{last.state.max_level if last.state.max_level is not None else '?'}; "
              f"{agent.action_counter} actions; stop: {agent.stop_reason}; session {agent.session_id}")
    print(f"artifacts: {workdir / 'world_model.py'}, {workdir / 'notes.md'}, {workdir / 'events.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
