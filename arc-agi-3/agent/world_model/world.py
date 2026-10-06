from __future__ import annotations

import builtins
import heapq
import inspect
import math
import time
from typing import Optional

import numpy as np

from .budgets import BFS_SEARCH_S


_ALLOWED_IMPORTS = {"numpy", "math", "collections", "itertools", "functools", "heapq"}

_SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in (
        "range", "len", "min", "max", "abs", "enumerate", "zip", "list", "dict",
        "set", "tuple", "int", "float", "bool", "str", "sum", "sorted", "any",
        "all", "map", "filter", "reversed", "round", "divmod", "pow", "isinstance",
        "getattr", "setattr", "hasattr", "Exception", "ValueError", "IndexError",
        "KeyError", "TypeError", "ZeroDivisionError", "slice", "frozenset", "bytes",
    )
}


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root not in _ALLOWED_IMPORTS:
        raise ImportError(f"import '{name}' is not allowed in world-model code")
    return __import__(name, globals, locals, fromlist, level)


def _exec_code(code: str) -> dict:
    ns: dict = {
        "__builtins__": {**_SAFE_BUILTINS, "__import__": _safe_import},
        "np": np,
        "numpy": np,
        "ENTRY_GRID": None,
        "CURRENT_LEVEL": None,
    }
    exec(compile(code, "<world_model>", "exec"), ns)
    return ns


def _to_list(grid) -> list:
    return np.asarray(grid).tolist()


def _arity(fn) -> int:
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return 1
    n = 0
    for p in params:
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            n += 1
        elif p.kind is p.VAR_POSITIONAL:
            return 99
    return n


class CodeWorldModel:

    def __init__(self, code: str) -> None:
        ns = _exec_code(code)
        predict = ns.get("predict")
        step = ns.get("step")
        init_state = ns.get("init_state")
        self.stateful = callable(predict)
        if not self.stateful and not callable(step):
            raise ValueError(
                "generated code must define predict(state, grid, action, x=None, y=None) "
                "OR step(grid, action, x=None, y=None)"
            )
        self.code = code
        self._ns = ns
        self._predict = predict if self.stateful else None
        self._step = step if callable(step) else None
        self._init_state = init_state if callable(init_state) else None

        def _pred(name):
            fn = ns.get(name)
            return (fn, _arity(fn)) if callable(fn) else (None, 0)
        self._win, self._win_arity = _pred("is_win_condition")
        self._bfs_goal, self._bfs_goal_arity = _pred("is_bfs_goal")
        self._legacy_goal, self._legacy_arity = _pred("is_goal")
        self._heur, self._heur_arity = _pred("bfs_heuristic")

    def set_entry_grid(self, grid, level: Optional[int] = None) -> None:
        self._ns["ENTRY_GRID"] = None if grid is None else _to_list(grid)
        self._ns["CURRENT_LEVEL"] = None if level is None else int(level)

    def init_state(self, entry_grid):
        if not self.stateful:
            return None
        if self._init_state is None:
            return {}
        eg = _to_list(entry_grid) if entry_grid is not None else None
        return self._init_state(eg)

    def predict(self, state, grid, action, x=None, y=None) -> "tuple[np.ndarray, dict, object]":
        if self.stateful:
            out = self._predict(state, _to_list(grid), int(action), x, y)
            return self._unpack(out, state)
        out = self._step(_to_list(grid), int(action), x, y)
        g, info, _ = self._unpack(out, None)
        return g, info, None

    @staticmethod
    def _unpack(out, prev_state) -> "tuple[np.ndarray, dict, object]":
        info: dict = {}
        next_state = prev_state
        if isinstance(out, tuple):
            grid = out[0]
            if len(out) > 1 and isinstance(out[1], dict):
                info = out[1]
            if len(out) > 2:
                next_state = out[2]
        else:
            grid = out
        return np.asarray(grid, dtype=np.int8), dict(info or {}), next_state

    def _call_pred(self, fn, arity, grid, state) -> bool:
        if arity >= 2:
            return bool(fn(state, _to_list(grid)))
        return bool(fn(_to_list(grid)))

    @property
    def has_win_condition(self) -> bool:
        return self._win is not None

    @property
    def has_heuristic(self) -> bool:
        return self._heur is not None

    def heuristic(self, grid, state=None) -> float:
        if self._heur is None:
            raise AttributeError("world model defines no bfs_heuristic")
        if self._heur_arity >= 2:
            return float(self._heur(state, _to_list(grid)))
        return float(self._heur(_to_list(grid)))

    @property
    def has_goal_pred(self) -> bool:
        return any(f is not None for f in (self._bfs_goal, self._win, self._legacy_goal))

    @property
    def has_advance_goal(self) -> bool:
        return self._win is not None or (self._bfs_goal is None and self._legacy_goal is not None)

    def win_condition(self, grid, state=None) -> Optional[bool]:
        if self._win is None:
            return None
        return self._call_pred(self._win, self._win_arity, grid, state)

    def goal_pred(self, grid, state=None) -> Optional[bool]:
        for fn, ar in ((self._bfs_goal, self._bfs_goal_arity),
                       (self._win, self._win_arity),
                       (self._legacy_goal, self._legacy_arity)):
            if fn is not None:
                return self._call_pred(fn, ar, grid, state)
        return None

    def advance_goal(self, grid, state=None) -> Optional[bool]:
        if self._win is not None:
            return self._call_pred(self._win, self._win_arity, grid, state)
        if self._bfs_goal is None and self._legacy_goal is not None:
            return self._call_pred(self._legacy_goal, self._legacy_arity, grid, state)
        return None


def _key(grid: np.ndarray) -> bytes:
    g = np.asarray(grid, dtype=np.int8)
    return g.shape[0].to_bytes(2, "little") + g.shape[1].to_bytes(2, "little") + g.tobytes()


def _node_key(grid: np.ndarray, state) -> bytes:
    k = _key(grid)
    if state is None:
        return k
    return k + b"|" + repr(state).encode("utf-8", "replace")


_MAIN_SHARE = 0.65
_MIN_FALLBACK_S = 5.0
_NOVELTY_COLORS = 256


class _Novelty:

    def __init__(self, grid) -> None:
        g = np.asarray(grid, dtype=np.int8)
        self.ok = g.ndim == 2
        self.shape = g.shape
        if self.ok:
            self.yy, self.xx = np.indices(g.shape)
            self.seen = np.zeros(g.shape + (_NOVELTY_COLORS,), dtype=bool)
            self.seen[self.yy, self.xx, g.astype(np.uint8)] = True

    def novel(self, grid) -> bool:
        if not self.ok:
            return True
        g = np.asarray(grid, dtype=np.int8)
        if g.shape != self.shape:
            return True
        c = g.astype(np.uint8)
        if self.seen[self.yy, self.xx, c].all():
            return False
        self.seen[self.yy, self.xx, c] = True
        return True


def _result(**kw) -> dict:
    r = {
        "plan": None, "goal_reason": None, "final_grid": None,
        "expanded": 0, "distinct": 0, "frontier": 0,
        "termination": "exhausted", "note": "",
        "algo": "none", "optimal": True, "best": None, "depth_reached": 0,
        "h_start": None, "heuristic_error": None, "predict_ms": None, "secs": 0.0,
        "phases": [], "time_budget_s": None, "has_heuristic": False,
    }
    r.update(kw)
    return r


def _search(world, start, start_state, moves, goal_reason, *, reset_grid, reset_state,
            max_depth, max_nodes, deadline, mode="bfs", h_fn=None, novelty=False) -> dict:
    t_start = time.monotonic()
    t_pred = 0.0
    n_pred = 0
    h_err: Optional[str] = None

    def H(g, st) -> float:
        nonlocal h_err
        if h_fn is None or h_err is not None:
            return 0.0
        try:
            v = float(h_fn(g, st))
            if not math.isfinite(v):
                raise ValueError(f"returned non-finite value {v!r}")
            return v if v > 0.0 else 0.0
        except Exception as e:
            h_err = f"{type(e).__name__}: {e}"
            return 0.0

    def prio(g_cost: int, h: float):
        if mode == "astar":
            return (g_cost + h, h)
        if mode == "greedy":
            return (h, g_cost)
        return (g_cost,)

    start = np.asarray(start, dtype=np.int8)
    h0 = H(start, start_state)
    best = (h0, 0, [], start) if h_fn is not None else None
    best_g: dict = {_node_key(start, start_state): 0}
    heap: list = []
    seq = 0
    heapq.heappush(heap, (prio(0, h0), seq, start, start_state, [], None, ""))
    seq += 1
    nov = _Novelty(start) if novelty else None
    expanded = 0
    depth_reached = 0
    hit_depth_cap = False
    timed_out = False
    algo = mode + ("+novelty" if novelty else "")

    def done(plan, reason, final, term, note="", optimal=True) -> dict:
        return _result(
            plan=plan, goal_reason=reason, final_grid=final,
            expanded=expanded, distinct=len(best_g), frontier=len(heap),
            termination=term, note=note, algo=algo, optimal=bool(optimal),
            best=None if best is None else
                 {"h": best[0], "depth": best[1], "path": list(best[2]), "grid": best[3]},
            depth_reached=depth_reached, h_start=h0 if h_fn is not None else None,
            heuristic_error=h_err,
            predict_ms=(1000.0 * t_pred / n_pred) if n_pred else None,
            secs=time.monotonic() - t_start,
        )

    def push(ng, nst, new_path, reason, note=""):
        nonlocal seq, depth_reached, best
        if reason is not None:
            if mode != "astar":
                return done(new_path, reason, ng, "found", note)
            heapq.heappush(heap, (prio(len(new_path), 0.0), seq, ng, nst, new_path, reason, note))
            seq += 1
            return None
        k = _node_key(ng, nst)
        prev = best_g.get(k)
        if prev is not None and prev <= len(new_path):
            return None
        if nov is not None and not nov.novel(ng):
            return None
        best_g[k] = len(new_path)
        h = H(ng, nst)
        if best is not None and (h, len(new_path)) < (best[0], best[1]):
            best = (h, len(new_path), new_path, ng)
        if len(new_path) > depth_reached:
            depth_reached = len(new_path)
        heapq.heappush(heap, (prio(len(new_path), h), seq, ng, nst, new_path, None, ""))
        seq += 1
        return None

    while heap and expanded < max_nodes:
        if time.monotonic() >= deadline:
            timed_out = True
            break
        _, _, g, st, path, reason, note = heapq.heappop(heap)
        if reason is not None:
            return done(path, reason, g, "found", note)
        if mode == "astar" and best_g.get(_node_key(g, st), len(path)) < len(path):
            continue
        if len(path) >= max_depth:
            hit_depth_cap = True
            continue
        for (a, x, y) in moves:
            expanded += 1
            t0 = time.perf_counter()
            try:
                ng, info, nst = world.predict(st, g, a, x, y)
            except Exception:
                t_pred += time.perf_counter() - t0
                continue
            t_pred += time.perf_counter() - t0
            n_pred += 1
            if info.get("dead"):
                continue
            new_path = path + [(a, x, y)]
            reason = goal_reason(info, ng, nst)
            if reason is None and (info.get("level_up") or info.get("win")):
                continue
            r = push(ng, nst, new_path, reason)
            if r is not None:
                return r
        if reset_grid is not None and not path:
            expanded += 1
            ng, nst = np.asarray(reset_grid, dtype=np.int8), reset_state
            r = push(ng, nst, [(0, None, None)], goal_reason({}, ng, nst),
                     note="RESET (level restart) already reaches the goal.")
            if r is not None:
                return r

    goals = [e for e in heap if e[5] is not None]
    if goals:
        e = min(goals, key=lambda e: (len(e[4]), e[1]))
        return done(e[4], e[5], e[2], "found", e[6], optimal=False)
    if timed_out:
        term = "timeout"
    elif expanded >= max_nodes:
        term = "budget"
    elif hit_depth_cap:
        term = "depth"
    else:
        term = "exhausted"
    return done(None, None, None, term)


def _phase(r: dict) -> dict:
    return {"algo": r["algo"], "termination": r["termination"], "expanded": r["expanded"],
            "distinct": r["distinct"], "secs": round(float(r["secs"]), 1)}


def bfs(
    world: CodeWorldModel,
    grid: np.ndarray,
    actions: list[int],
    *,
    clicks: Optional[list[tuple[int, int]]] = None,
    target: str = "advance",
    start_state=None,
    reset_grid: Optional[np.ndarray] = None,
    reset_state=None,
    max_depth: int = 50,
    max_nodes: int = 20000,
    time_budget_s: float = BFS_SEARCH_S,
) -> dict:
    pred = None
    pred_reason = ""
    if target == "is_goal" and getattr(world, "has_goal_pred", False):
        pred, pred_reason = world.goal_pred, "is_goal"
    elif target == "advance" and getattr(world, "has_advance_goal", False):
        pred, pred_reason = world.advance_goal, "win_condition"
    h_fn = world.heuristic if getattr(world, "has_heuristic", False) else None
    if target == "is_goal" and pred is None:
        return _result(
            distinct=0, termination="no_goal_capability", time_budget_s=time_budget_s,
            has_heuristic=h_fn is not None,
            note="target='is_goal' but the model defines no goal predicate "
                 "(is_bfs_goal / is_win_condition / legacy is_goal).",
        )

    moves: list[tuple[int, Optional[int], Optional[int]]] = [(int(a), None, None) for a in actions]
    for (cx, cy) in (clicks or []):
        moves.append((6, int(cx), int(cy)))

    def goal_reason(info: dict, ng: np.ndarray, nst) -> Optional[str]:
        if info.get("win"):
            return "win"
        if info.get("level_up"):
            return "level_up" if target in ("advance", "level_up") else None
        if pred is not None and bool(pred(ng, nst)):
            return pred_reason
        return None

    start = np.asarray(grid, dtype=np.int8)
    if pred is not None and bool(pred(start, start_state)):
        return _result(
            plan=[], goal_reason=pred_reason, final_grid=start, distinct=1,
            termination="found", time_budget_s=time_budget_s, has_heuristic=h_fn is not None,
            note="start grid already satisfies the goal predicate.",
        )

    t0 = time.monotonic()
    common = dict(reset_grid=reset_grid, reset_state=reset_state,
                  max_depth=max_depth, max_nodes=max_nodes, h_fn=h_fn)
    r = _search(world, start, start_state, moves, goal_reason,
                mode="astar" if h_fn is not None else "bfs",
                deadline=t0 + time_budget_s * _MAIN_SHARE, **common)
    phases = [_phase(r)]
    out = r
    if r["plan"] is None and r["termination"] in ("budget", "timeout"):
        remaining = t0 + time_budget_s - time.monotonic()
        if remaining >= _MIN_FALLBACK_S:
            r2 = _search(world, start, start_state, moves, goal_reason,
                         mode="greedy" if h_fn is not None else "bfs", novelty=h_fn is None,
                         deadline=t0 + time_budget_s, **common)
            phases.append(_phase(r2))
            if r2["plan"] is not None:
                out = r2
                out["optimal"] = False
            else:
                b1, b2 = r["best"], r2["best"]
                if b1 is not None and b2 is not None and (b2["h"], b2["depth"]) < (b1["h"], b1["depth"]):
                    r["best"] = b2
                r["depth_reached"] = max(r["depth_reached"], r2["depth_reached"])
                r["heuristic_error"] = r["heuristic_error"] or r2["heuristic_error"]
                if r["predict_ms"] is None:
                    r["predict_ms"] = r2["predict_ms"]
            out["expanded"] = r["expanded"] + r2["expanded"]
    out["phases"] = phases
    out["secs"] = time.monotonic() - t0
    out["time_budget_s"] = time_budget_s
    out["has_heuristic"] = h_fn is not None
    return out


def rollout_state(world, timeline, entry_grids, level, end):
    if world is None:
        return None
    entry = entry_grids.get(int(level))
    try:
        world.set_entry_grid(entry, level)
        state = world.init_state(entry)
        if not world.stateful:
            return None
    except Exception:
        return None
    for i in range(end):
        t = timeline[i]
        if t.before_levels != level:
            continue
        if t.action_id == 0:
            try:
                state = world.init_state(entry)
            except Exception:
                return None
            continue
        before = timeline[i - 1].after if i >= 1 else None
        if before is None:
            continue
        try:
            _, _, state = world.predict(state, before, t.action_id, t.x, t.y)
        except Exception:
            break
    return state


def _score_step(world, ts, after, terminal, pred, info, next_state, errors: list, kinds: list) -> None:
    for flag in ("level_up", "dead", "win"):
        pf, af = bool(info.get(flag)), bool(getattr(ts, flag))
        if pf != af:
            errors.append(f"{flag} prediction error: predicted {pf}, actual {af}")
            kinds.append(flag)
    if not terminal and after is not None:
        if pred.shape != after.shape:
            errors.append(
                f"grid prediction error: predicted shape {tuple(pred.shape)} "
                f"!= actual {tuple(after.shape)}"
            )
            kinds.append("grid")
        elif not np.array_equal(pred, after):
            errors.append(f"grid prediction error: {int((pred != after).sum())} cells differ from actual")
            kinds.append("grid")
    if getattr(world, "has_win_condition", False):
        try:
            if not terminal:
                if after is not None and world.win_condition(after, next_state):
                    errors.append(
                        "win_condition error: is_win_condition says this reached state "
                        "completes the level, but in reality it did not (no level_up)")
                    kinds.append("win_cond")
            elif (ts.level_up or ts.win) and not errors:
                if pred is not None and not world.win_condition(pred, next_state):
                    errors.append(
                        "win_condition incoherence: your flags predict this step completes "
                        "the level, but is_win_condition(your own predicted next grid) is False "
                        "— the flag and the predicate must encode the same rule")
                    kinds.append("win_cond")
        except Exception as e:
            errors.append(f"is_win_condition raised {type(e).__name__}: {e}")
            kinds.append("win_cond")


def _heuristic_report(world, before, state) -> "tuple[float | None, str | None]":
    if not getattr(world, "has_heuristic", False):
        return None, None
    try:
        h_val = float(world.heuristic(before, state))
        if not math.isfinite(h_val):
            raise ValueError(f"returned non-finite value {h_val!r}")
        return h_val, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def backtest_rollout(world, timeline, entry_grids):
    results: dict = {}
    cur_level = None
    state = None
    for i, ts in enumerate(timeline):
        L = ts.before_levels
        if L != cur_level:
            cur_level = L
            world.set_entry_grid(entry_grids.get(L), L)
            state = world.init_state(entry_grids.get(L))
        if ts.action_id == 0:
            state = world.init_state(entry_grids.get(L))
            continue
        before = timeline[i - 1].after if i >= 1 else None
        if before is None:
            continue
        after = ts.after
        terminal = ts.level_up or ts.dead or ts.win
        errors: list = []
        kinds: list = []
        pred = None
        info: dict = {}
        world.set_entry_grid(entry_grids.get(L), L)
        h_val, h_err = _heuristic_report(world, before, state)
        t_pred = time.perf_counter()
        try:
            pred, info, next_state = world.predict(state, before, ts.action_id, ts.x, ts.y)
        except Exception as e:
            errors.append(f"predict() raised {type(e).__name__}: {e}")
            kinds.append("error")
        else:
            _score_step(world, ts, after, terminal, pred, info, next_state, errors, kinds)
            state = next_state
        results[i] = {"pred": pred, "info": info, "errors": errors,
                      "kinds": kinds, "terminal": terminal, "before": before, "after": after,
                      "h": h_val, "h_error": h_err,
                      "predict_ms": 1000.0 * (time.perf_counter() - t_pred)}
    return results
