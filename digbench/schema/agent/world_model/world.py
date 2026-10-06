from __future__ import annotations

import builtins
import inspect
import re
import time
from collections import deque
from typing import Optional

import numpy as np

_ALLOWED_IMPORTS = {
    "numpy", "math", "collections", "itertools", "functools", "heapq", "re", "string",
    "copy", "dataclasses", "typing", "enum", "operator", "bisect",
}

_SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in (
        "range", "len", "min", "max", "abs", "enumerate", "zip", "list", "dict",
        "set", "tuple", "int", "float", "bool", "str", "sum", "sorted", "any",
        "all", "map", "filter", "reversed", "round", "divmod", "pow", "isinstance",
        "getattr", "setattr", "hasattr", "Exception", "ValueError", "IndexError",
        "KeyError", "TypeError", "ZeroDivisionError", "slice", "frozenset", "bytes",
        "chr", "ord", "repr", "hash", "type", "iter", "next", "callable", "object",
        "StopIteration", "AttributeError", "RuntimeError", "AssertionError", "NotImplementedError",
        "property", "staticmethod", "classmethod", "super", "id", "format", "bytearray",
    )
}

FLAG_NAMES = ("level_up", "life_lost", "dead", "win")


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
        "ENTRY_OBS": None,
        "CURRENT_LEVEL": None,
        "CURRENT_MODE": None,
    }
    exec(compile(code, "<world_model>", "exec"), ns)
    return ns


def _to_text(obs) -> str:
    if obs is None:
        return ""
    if isinstance(obs, str):
        return obs
    if isinstance(obs, (list, tuple)) and all(isinstance(x, str) for x in obs):
        return "\n".join(obs)
    return str(obs)


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


def diff_summary(pred: Optional[str], actual: Optional[str]) -> str:
    if pred is None or actual is None:
        return "no prediction" if pred is None else "no actual observation"
    if pred == actual:
        return "identical"
    if len(pred) != len(actual):
        head = f"length {len(pred)} (predicted) vs {len(actual)} (actual)"
    else:
        head = f"same length {len(actual)}"
    n = min(len(pred), len(actual))
    first = next((i for i in range(n) if pred[i] != actual[i]), n)
    diffs = sum(1 for i in range(n) if pred[i] != actual[i]) + abs(len(pred) - len(actual))
    return (f"{head}; {diffs} char position(s) differ; first difference at index {first} "
            f"(predicted {pred[first:first + 8]!r} vs actual {actual[first:first + 8]!r})")


class CodeWorldModel:

    def __init__(self, code: str) -> None:
        ns = _exec_code(code)
        predict = ns.get("predict")
        step = ns.get("step")
        self.stateful = callable(predict)
        if not self.stateful and not callable(step):
            raise ValueError(
                "world-model code must define predict(state, obs, action) OR step(obs, action)"
            )
        self.code = code
        self._ns = ns
        self._predict = predict if self.stateful else None
        self._step = step if callable(step) else None
        self._init_state = ns.get("init_state") if callable(ns.get("init_state")) else None

        def _pred(name):
            fn = ns.get(name)
            return (fn, _arity(fn)) if callable(fn) else (None, 0)

        self._win, self._win_arity = _pred("is_win_condition")
        self._bfs_goal, self._bfs_goal_arity = _pred("is_bfs_goal")
        self._legacy_goal, self._legacy_arity = _pred("is_goal")

    def set_entry(self, entry: Optional[dict]) -> None:
        entry = entry or {}
        self._ns["ENTRY_OBS"] = None if entry.get("obs") is None else _to_text(entry.get("obs"))
        lv = entry.get("level")
        self._ns["CURRENT_LEVEL"] = None if lv is None else int(lv)
        self._ns["CURRENT_MODE"] = entry.get("mode")

    def init_state(self, entry_obs):
        if not self.stateful:
            return None
        if self._init_state is None:
            return {}
        return self._init_state(None if entry_obs is None else _to_text(entry_obs))

    def predict(self, state, obs, action) -> "tuple[str, dict, object]":
        if self.stateful:
            out = self._predict(state, _to_text(obs), str(action))
            return self._unpack(out, state)
        out = self._step(_to_text(obs), str(action))
        o, info, _ = self._unpack(out, None)
        return o, info, None

    @staticmethod
    def _unpack(out, prev_state) -> "tuple[str, dict, object]":
        info: dict = {}
        next_state = prev_state
        if isinstance(out, tuple):
            obs = out[0] if out else ""
            if len(out) > 1 and isinstance(out[1], dict):
                info = out[1]
            if len(out) > 2:
                next_state = out[2]
        else:
            obs = out
        return _to_text(obs), dict(info or {}), next_state

    def step(self, obs, action) -> "tuple[str, dict]":
        st = self.init_state(None) if self.stateful else None
        o, info, _ = self.predict(st, obs, action)
        return o, info

    @staticmethod
    def _call_pred(fn, arity, obs, state) -> bool:
        if arity >= 2:
            return bool(fn(state, _to_text(obs)))
        return bool(fn(_to_text(obs)))

    @property
    def has_win_condition(self) -> bool:
        return self._win is not None

    @property
    def has_goal_pred(self) -> bool:
        return any(f is not None for f in (self._bfs_goal, self._win, self._legacy_goal))

    @property
    def has_advance_goal(self) -> bool:
        return self._win is not None or (self._bfs_goal is None and self._legacy_goal is not None)

    def win_condition(self, obs, state=None) -> Optional[bool]:
        if self._win is None:
            return None
        return self._call_pred(self._win, self._win_arity, obs, state)

    def goal_pred(self, obs, state=None) -> Optional[bool]:
        for fn, ar in ((self._bfs_goal, self._bfs_goal_arity),
                       (self._win, self._win_arity),
                       (self._legacy_goal, self._legacy_arity)):
            if fn is not None:
                return self._call_pred(fn, ar, obs, state)
        return None

    def advance_goal(self, obs, state=None) -> Optional[bool]:
        if self._win is not None:
            return self._call_pred(self._win, self._win_arity, obs, state)
        if self._bfs_goal is None and self._legacy_goal is not None:
            return self._call_pred(self._legacy_goal, self._legacy_arity, obs, state)
        return None


_HIST_PREV_RE = re.compile(r"\(previous\)[ \t]*$", re.M)
_HIST_CUR_RE = re.compile(r"\(current\)[ \t]*$", re.M)


def has_history(obs) -> bool:
    if obs is None:
        return False
    o = _to_text(obs)
    return bool(_HIST_PREV_RE.search(o)) and bool(_HIST_CUR_RE.search(o))


def split_history(obs) -> "tuple[str, str]":
    o = _to_text(obs)
    if not has_history(o):
        return "", o
    i = o.rfind("\n\n")
    if i < 0:
        return "", o
    return o[: i + 2], o[i + 2:]


def current_frame(obs) -> str:
    return split_history(obs)[1]


def frames_match(pred, actual) -> Optional[str]:
    if pred is None or actual is None:
        return None
    if pred == actual:
        return "exact"
    if has_history(actual) and current_frame(pred) == current_frame(actual):
        return "current"
    return None


def _node_key(obs: str, state) -> str:
    key = current_frame(obs)
    if state is None:
        return key
    return key + "\x00" + repr(state)


def bfs(
    world: CodeWorldModel,
    obs: str,
    actions: list[str],
    *,
    target: str = "advance",
    start_state=None,
    max_depth: int = 20,
    max_nodes: int = 20000,
    time_budget_s: float = 600.0,
) -> dict:
    pred = None
    pred_reason = ""
    if target == "is_goal" and getattr(world, "has_goal_pred", False):
        pred, pred_reason = world.goal_pred, "is_goal"
    elif target == "advance" and getattr(world, "has_advance_goal", False):
        pred, pred_reason = world.advance_goal, "win_condition"
    if target == "is_goal" and pred is None:
        return {
            "plan": None, "goal_reason": None, "final_obs": None,
            "expanded": 0, "distinct": 0, "frontier": 0,
            "termination": "no_goal_capability",
            "note": "target='is_goal' but the model defines no goal predicate "
                    "(is_bfs_goal / is_win_condition / legacy is_goal).",
        }

    moves = [str(a) for a in actions]

    def goal_reason(info: dict, no: str, nst) -> Optional[str]:
        if info.get("win"):
            return "win"
        if info.get("level_up"):
            return "level_up" if target in ("advance", "level_up") else None
        if pred is not None and bool(pred(no, nst)):
            return pred_reason
        return None

    start = _to_text(obs)
    if pred is not None and bool(pred(start, start_state)):
        return {
            "plan": [], "goal_reason": pred_reason, "final_obs": start,
            "expanded": 0, "distinct": 1, "frontier": 0,
            "termination": "found", "note": "start observation already satisfies the goal predicate.",
        }

    queue: deque = deque([(start, start_state, [])])
    seen = {_node_key(start, start_state)}
    expanded = 0
    hit_depth_cap = False
    timed_out = False
    deadline = time.monotonic() + time_budget_s

    while queue and expanded < max_nodes:
        if time.monotonic() >= deadline:
            timed_out = True
            break
        o, st, path = queue.popleft()
        if len(path) >= max_depth:
            hit_depth_cap = True
            continue
        for a in moves:
            expanded += 1
            try:
                no, info, nst = world.predict(st, o, a)
            except Exception:
                continue
            new_path = path + [a]
            if info.get("dead") or info.get("life_lost"):
                continue
            reason = goal_reason(info, no, nst)
            if reason is not None:
                return {
                    "plan": new_path, "goal_reason": reason, "final_obs": no,
                    "expanded": expanded, "distinct": len(seen), "frontier": len(queue),
                    "termination": "found", "note": "",
                }
            if info.get("level_up") or info.get("win"):
                continue
            k = _node_key(no, nst)
            if k in seen:
                continue
            seen.add(k)
            queue.append((no, nst, new_path))

    if timed_out:
        termination = "timeout"
    elif expanded >= max_nodes:
        termination = "budget"
    elif hit_depth_cap:
        termination = "depth"
    else:
        termination = "exhausted"
    return {
        "plan": None, "goal_reason": None, "final_obs": None,
        "expanded": expanded, "distinct": len(seen), "frontier": len(queue),
        "termination": termination, "note": "",
    }


def _checkable(ts) -> bool:
    scored = getattr(ts, "scored", None)
    if scored is not None:
        return bool(scored)
    return not getattr(ts, "invalid", False) and not getattr(ts, "mode_switch", False)


def rollout_state(world, timeline, entries: dict, segment: int, end: int):
    if world is None:
        return None
    entry = entries.get(int(segment))
    try:
        world.set_entry(entry)
        state = world.init_state(None if entry is None else entry.get("obs"))
        if not world.stateful:
            return None
    except Exception:
        return None
    for i in range(min(end, len(timeline))):
        t = timeline[i]
        if int(t.segment) != int(segment) or not _checkable(t):
            continue
        if t.before_obs is None:
            continue
        try:
            _, _, state = world.predict(state, t.before_obs, t.action)
        except Exception:
            break
    return state


def flag_mismatches(info: dict, ts) -> "tuple[list[str], list[str]]":
    errors: list[str] = []
    kinds: list[str] = []
    lenient_budget = bool(getattr(ts, "budget_exhausted", False))
    challenge = getattr(ts, "challenge_outcome", None)
    p_level_up = bool(info.get("level_up"))
    p_life_lost = bool(info.get("life_lost"))
    for flag in FLAG_NAMES:
        pf, af = bool(info.get(flag)), bool(getattr(ts, flag))
        if pf == af:
            continue
        if pf and not af and challenge is not None:
            if challenge == "solved" and flag in ("level_up", "win"):
                continue
            if challenge == "failed" and flag in ("life_lost", "dead"):
                continue
        if af and not pf:
            if flag == "win" and p_level_up:
                continue
            if flag == "dead" and (p_life_lost or lenient_budget):
                continue
            if flag == "life_lost" and lenient_budget:
                continue
        errors.append(f"{flag} prediction error: predicted {pf}, actual {af}")
        kinds.append(flag)
    return errors, kinds


def score_step(world, pred, info: dict, next_state, ts) -> dict:
    errors: list[str] = []
    kinds: list[str] = []
    warnings: list[str] = []
    win_verified = False
    terminal = bool(getattr(ts, "terminal", False))
    after = getattr(ts, "after", None)
    final = getattr(ts, "final_frame", None)
    challenge = getattr(ts, "challenge_outcome", None)
    fe, fk = flag_mismatches(info or {}, ts)
    errors += fe
    kinds += fk
    target = final if terminal else after
    what = "final-board" if terminal else "observation"
    if target is not None:
        m = frames_match(pred, target)
        if m is None:
            errors.append(f"{what} prediction error: " + diff_summary(pred, target))
            kinds.append("obs")
        elif m == "current":
            warnings.append(f"{what}: current frame correct, embedded history prefix differs "
                            "(not counted as a mismatch)")
            kinds.append("obs_hist")
    if getattr(world, "has_win_condition", False) and hasattr(world, "win_condition"):
        positive = bool(getattr(ts, "level_up", False) or getattr(ts, "win", False) or challenge == "solved")
        lost_by_state = bool(challenge == "failed" or (getattr(ts, "life_lost", False)
                                                      and not getattr(ts, "budget_exhausted", False)))
        try:
            if not terminal:
                if after is not None and world.win_condition(after, next_state):
                    errors.append(
                        "win_condition error: is_win_condition says this reached state "
                        "completes the level, but in reality it did not (no level_up)")
                    kinds.append("win_cond")
            elif final is not None and positive:
                if world.win_condition(final, next_state):
                    win_verified = True
                else:
                    label = ("the creative challenge" if challenge == "solved"
                             else f"level {getattr(ts, 'level_before', '?')}")
                    errors.append(
                        f"win_condition error: is_win_condition is False on the REAL final board "
                        f"that completed {label} (the bench reported it) — your win rule does not "
                        "explain this actual completion")
                    kinds.append("win_cond")
            elif final is not None and lost_by_state:
                if world.win_condition(final, next_state):
                    errors.append(
                        "win_condition error: is_win_condition is True on the REAL final board of "
                        "a LOST level / failed challenge — that board did not complete anything")
                    kinds.append("win_cond")
            elif positive and pred is not None and not errors:
                if not world.win_condition(pred, next_state):
                    errors.append(
                        "win_condition incoherence: your flags predict this step completes "
                        "the level, but is_win_condition(your own predicted next observation) "
                        "is False — the flag and the predicate must encode the same rule")
                    kinds.append("win_cond")
        except Exception as e:
            errors.append(f"is_win_condition raised {type(e).__name__}: {e}")
            kinds.append("win_cond")
    return {"errors": errors, "kinds": kinds, "warnings": warnings, "win_verified": win_verified}


def backtest_rollout(world, timeline, entries: dict) -> dict:
    results: dict = {}
    cur_seg = None
    state = None
    for i, ts in enumerate(timeline):
        if not _checkable(ts):
            continue
        seg = int(ts.segment)
        entry = entries.get(seg)
        if seg != cur_seg:
            cur_seg = seg
            try:
                world.set_entry(entry)
                state = world.init_state(None if entry is None else entry.get("obs"))
            except Exception as e:
                results[i] = {"pred": None, "info": {}, "errors": [f"init_state raised {type(e).__name__}: {e}"],
                              "kinds": ["error"], "warnings": [], "win_verified": False,
                              "terminal": bool(ts.terminal), "before": ts.before_obs, "after": ts.after,
                              "final_frame": getattr(ts, "final_frame", None)}
                state = None
                continue
        before = ts.before_obs
        after = ts.after
        terminal = bool(ts.terminal)
        errors: list = []
        kinds: list = []
        warnings: list = []
        win_verified = False
        pred = None
        info: dict = {}
        world.set_entry(entry)
        try:
            pred, info, next_state = world.predict(state, before, ts.action)
        except Exception as e:
            errors.append(f"predict() raised {type(e).__name__}: {e}")
            kinds.append("error")
        else:
            sc = score_step(world, pred, info, next_state, ts)
            errors += sc["errors"]
            kinds += sc["kinds"]
            warnings += sc["warnings"]
            win_verified = sc["win_verified"]
            state = next_state
        results[i] = {"pred": pred, "info": info, "errors": errors, "kinds": kinds,
                      "warnings": warnings, "win_verified": win_verified, "terminal": terminal,
                      "before": before, "after": after, "final_frame": getattr(ts, "final_frame", None)}
    return results
