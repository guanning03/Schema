from __future__ import annotations

from typing import Optional

import numpy as np


def ser_grid(g) -> Optional[list]:
    if g is None:
        return None
    return g.tolist() if hasattr(g, "tolist") else g


def ser_timeline(timeline, end: Optional[int] = None) -> list:
    end = len(timeline) if end is None else end
    out = []
    for t in timeline[:end]:
        out.append({
            "action_id": t.action_id, "x": t.x, "y": t.y,
            "before_levels": t.before_levels, "after": ser_grid(t.after),
            "level_up": bool(t.level_up), "dead": bool(t.dead), "win": bool(t.win),
        })
    return out


def ser_entry(entry_grids: dict) -> dict:
    return {str(k): ser_grid(v) for k, v in entry_grids.items()}


class WorldModelProxy:
    def __init__(self, executor, *, stateful: bool, has_is_goal: bool,
                 has_win_condition: bool = False) -> None:
        self._ex = executor
        self.stateful = bool(stateful)
        self.has_is_goal = bool(has_is_goal)
        self.has_win_condition = bool(has_win_condition)

    def set_entry_grid(self, grid, level: Optional[int] = None) -> None:
        self._ex.world_set_entry(grid=ser_grid(grid), level=level)

    def predict_step(self, timeline, level, entry_grids, end, before_grid, action, x, y,
                     want_state: bool = False):
        r = self._ex.world_predict_step(
            timeline=ser_timeline(timeline, end), entry=ser_entry(entry_grids), level=level,
            before_grid=ser_grid(before_grid), action=int(action), x=x, y=y,
            want_state=want_state)
        if not isinstance(r, dict) or "grid" not in r:
            return None
        grid, info = np.asarray(r["grid"], dtype=np.int8), (r.get("info") or {})
        return (grid, info, r.get("state_repr")) if want_state else (grid, info)

    def bfs(self, grid, acts, clicks, target, timeline, entry_grids, level,
            max_depth, max_nodes, allow_reset):
        r = self._ex.world_bfs(
            grid=ser_grid(grid), acts=list(acts), clicks=[list(c) for c in clicks],
            target=target, timeline=ser_timeline(timeline), entry=ser_entry(entry_grids),
            level=level, max_depth=max_depth, max_nodes=max_nodes, allow_reset=allow_reset)
        if r.get("plan") is not None:
            r["plan"] = [tuple(p) for p in r["plan"]]
        fg = r.get("final_grid")
        if fg is not None:
            r["final_grid"] = np.asarray(fg, dtype=np.int8)
        b = r.get("best")
        if b is not None:
            b = dict(b)
            if b.get("grid") is not None:
                b["grid"] = np.asarray(b["grid"], dtype=np.int8)
            b["path"] = [tuple(p) for p in (b.get("path") or [])]
            r["best"] = b
        return r

    def backtest(self, timeline, entry_grids, code: Optional[str] = None) -> dict:
        r = self._ex.world_backtest(timeline=ser_timeline(timeline), entry=ser_entry(entry_grids),
                                    code=code)
        out: dict = {}
        for k, v in (r.get("results") or {}).items():
            out[int(k)] = {
                "errors": v["errors"], "kinds": v["kinds"], "terminal": bool(v["terminal"]),
                "info": v.get("info") or {},
                "before": None if v["before"] is None else np.asarray(v["before"], dtype=np.int8),
                "after": None if v["after"] is None else np.asarray(v["after"], dtype=np.int8),
                "pred": None if v["pred"] is None else np.asarray(v["pred"], dtype=np.int8),
                "h": v.get("h"), "h_error": v.get("h_error"), "predict_ms": v.get("predict_ms"),
            }
        return {"results": out, "stateful": bool(r.get("stateful"))}
