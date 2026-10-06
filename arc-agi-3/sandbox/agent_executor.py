from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

_SELF = Path(__file__).resolve()
_LANDLOCK = _SELF.parent.parent / "agent" / "world_model" / "landlock_exec.py"
_CLIP = 30_000


def _landlock_ok() -> bool:
    try:
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        libc.syscall.restype = ctypes.c_long
        return libc.syscall(444, None, 0, 1) >= 1
    except Exception:
        return False


class _Step:
    __slots__ = ("action_id", "x", "y", "before_levels", "after", "level_up", "dead", "win")

    def __init__(self, d: dict) -> None:
        import numpy as np
        self.action_id = int(d["action_id"])
        self.x = d.get("x")
        self.y = d.get("y")
        self.before_levels = d["before_levels"]
        a = d.get("after")
        self.after = np.asarray(a, dtype=np.int8) if a is not None else None
        self.level_up = bool(d.get("level_up"))
        self.dead = bool(d.get("dead"))
        self.win = bool(d.get("win"))


def _mk_steps(dicts: list) -> list:
    return [_Step(d) for d in dicts]


def _mk_entry(d: dict) -> dict:
    import numpy as np
    return {int(k): (np.asarray(v, dtype=np.int8) if v is not None else None) for k, v in d.items()}


def _flags(info) -> dict:
    return {k: bool((info or {}).get(k)) for k in ("level_up", "dead", "win")}


def _grid(g):
    return None if g is None else (g.tolist() if hasattr(g, "tolist") else g)


class Executor:
    def __init__(self, workdir: str, allow_only: list[str]) -> None:
        self.workdir = Path(workdir)
        self.allow_only = [p for p in allow_only if p]
        self._py = sys.executable or "python3"
        self._world = None

    def _wrap(self, cmd: list[str]) -> list[str]:
        if not (self.allow_only and _landlock_ok()):
            return cmd
        wd = str(self.workdir.resolve())
        return [self._py, str(_LANDLOCK), "--allow-only", wd, *self.allow_only, "--", *cmd]

    def _exec(self, cmd: list[str], *, timeout: float, stdin: "str | None" = None,
              cwd: "str | None" = None) -> dict:
        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                self._wrap(cmd), cwd=str(cwd or self.workdir), input=stdin,
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            return {"returncode": -1, "seconds": time.monotonic() - t0,
                    "stdout": self._clip(e.stdout if isinstance(e.stdout, str) else ""),
                    "stderr": self._clip(e.stderr if isinstance(e.stderr, str) else ""),
                    "timed_out": True, "timeout": timeout}
        except FileNotFoundError as e:
            return {"returncode": 127, "seconds": 0.0, "stdout": "", "stderr": f"cannot exec: {e}"}
        return {"returncode": proc.returncode, "seconds": time.monotonic() - t0,
                "stdout": self._clip(proc.stdout), "stderr": self._clip(proc.stderr)}

    @staticmethod
    def _clip(s: str) -> str:
        if len(s) <= _CLIP:
            return s
        half = _CLIP // 2
        return s[:half] + f"\n… [clipped {len(s) - _CLIP} chars] …\n" + s[-half:]

    def op_ping(self, req: dict) -> dict:
        return {"pong": True, "uid": os.getuid(), "cwd": str(self.workdir)}

    def op_run_python(self, req: dict) -> dict:
        code, path = req.get("code"), req.get("path")
        argv = [str(a) for a in (req.get("args") or [])]
        stdin = req.get("stdin") if isinstance(req.get("stdin"), str) else None
        timeout = float(req.get("timeout", 300))
        if path:
            cmd = [self._py, str(path), *argv]
        elif isinstance(code, str) and code.strip():
            cmd = [self._py, "-c", code, *argv]
        else:
            return {"returncode": 2, "seconds": 0.0, "stdout": "",
                    "stderr": "run_python needs 'code' or 'path'."}
        return self._exec(cmd, timeout=timeout, stdin=stdin)

    def op_run_shell(self, req: dict) -> dict:
        command = req.get("command")
        cwd = req.get("cwd") or None
        timeout = float(req.get("timeout", 300))
        if not isinstance(command, str) or not command.strip():
            return {"returncode": 2, "seconds": 0.0, "stdout": "",
                    "stderr": "run_shell needs 'command'."}
        shell = shutil.which("bash") or "/bin/bash"
        return self._exec([shell, "-c", command], timeout=timeout, cwd=cwd)

    def op_world_load(self, req: dict) -> dict:
        from agent.world_model.world import CodeWorldModel
        try:
            self._world = CodeWorldModel(req["code"])
        except Exception as e:
            self._world = None
            return {"loaded": False, "error": f"{type(e).__name__}: {e}"}
        w = self._world
        return {"loaded": True, "stateful": bool(w.stateful),
                "has_is_goal": bool(w.has_goal_pred),
                "has_win_condition": bool(w.has_win_condition),
                "has_heuristic": bool(w.has_heuristic)}

    def op_world_set_entry(self, req: dict) -> dict:
        if self._world is None:
            return {"error": "no world model loaded"}
        self._world.set_entry_grid(req.get("grid"), req.get("level"))
        return {"ok": True}

    def op_world_predict_step(self, req: dict) -> dict:
        from agent.world_model.world import rollout_state
        w = self._world
        if w is None:
            return {"error": "no world model loaded"}
        tl = _mk_steps(req["timeline"])
        eg = _mk_entry(req["entry"])
        level = req["level"]
        try:
            state = rollout_state(w, tl, eg, level, len(tl))
            w.set_entry_grid(eg.get(int(level)), level)
            pred, info, nxt = w.predict(state, req["before_grid"], int(req["action"]),
                                        req.get("x"), req.get("y"))
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}
        out = {"grid": pred.tolist(), "info": _flags(info)}
        if req.get("want_state"):
            try:
                r = repr(nxt)
            except Exception as e:
                r = f"<repr failed: {type(e).__name__}>"
            out["state_repr"] = r[:4000]
        return out

    def op_world_bfs(self, req: dict) -> dict:
        from agent.world_model.world import bfs, rollout_state
        w = self._world
        if w is None:
            return {"error": "no world model loaded"}
        tl = _mk_steps(req["timeline"])
        eg = _mk_entry(req["entry"])
        level = req["level"]
        entry = eg.get(int(level))
        acts = [int(a) for a in req["acts"]]
        clicks = [(int(a), int(b)) for a, b in req.get("clicks", [])]
        w.set_entry_grid(entry, level)
        start_state = rollout_state(w, tl, eg, level, len(tl))
        allow_reset = bool(req.get("allow_reset", True))
        reset_grid = entry if (allow_reset and entry is not None) else None
        reset_state = None
        if reset_grid is not None:
            try:
                reset_state = w.init_state(entry)
            except Exception:
                reset_grid = None
        r = bfs(w, req["grid"], acts, clicks=clicks, target=req["target"],
                start_state=start_state, reset_grid=reset_grid, reset_state=reset_state,
                max_depth=int(req["max_depth"]), max_nodes=int(req["max_nodes"]),
                time_budget_s=float(req["time_budget_s"]))
        r = dict(r)
        r["final_grid"] = _grid(r.get("final_grid"))
        if r.get("plan") is not None:
            r["plan"] = [[a, x, y] for (a, x, y) in r["plan"]]
        if r.get("best") is not None:
            b = dict(r["best"])
            b["grid"] = _grid(b.get("grid"))
            b["path"] = [[a, x, y] for (a, x, y) in (b.get("path") or [])]
            r["best"] = b
        return r

    def op_world_backtest(self, req: dict) -> dict:
        from agent.world_model.world import CodeWorldModel, backtest_rollout
        code = req.get("code")
        if code:
            try:
                w = CodeWorldModel(code)
            except Exception as e:
                return {"error": f"{type(e).__name__}: {e}"}
        else:
            w = self._world
            if w is None:
                return {"error": "no world model loaded"}
        tl = _mk_steps(req["timeline"])
        eg = _mk_entry(req["entry"])
        results = backtest_rollout(w, tl, eg)
        out = {}
        for i, r in results.items():
            out[str(i)] = {"errors": r["errors"], "kinds": r["kinds"],
                           "terminal": bool(r["terminal"]), "info": _flags(r["info"]),
                           "before": _grid(r["before"]), "after": _grid(r["after"]),
                           "pred": _grid(r["pred"]),
                           "h": r.get("h"), "h_error": r.get("h_error"),
                           "predict_ms": r.get("predict_ms")}
        return {"results": out, "stateful": bool(w.stateful)}

    def dispatch(self, req: dict) -> dict:
        op = req.get("op", "")
        fn = getattr(self, f"op_{op}", None)
        if fn is None:
            return {"id": req.get("id"), "ok": False, "error": f"unknown op: {op!r}"}
        try:
            return {"id": req.get("id"), "ok": True, "result": fn(req)}
        except Exception as e:
            return {"id": req.get("id"), "ok": False, "error": f"{type(e).__name__}: {e}"}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--workdir", required=True)
    p.add_argument("--allow-only", default="")
    args = p.parse_args()

    ex = Executor(args.workdir, [s for s in args.allow_only.split(":") if s])
    sys.stderr.write(f"executor: up uid={os.getuid()} workdir={args.workdir}\n")
    sys.stderr.flush()

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = ex.dispatch(req)
        sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
