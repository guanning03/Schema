from __future__ import annotations

import builtins
import copy
import hashlib
import heapq
import inspect
import math
import os
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

FLAG_NAMES = ("gem", "room_changed", "dead", "won")

VIEW_NAMES = ("top", "top-diagonal", "diagonal", "side-diagonal", "side")

_ALLOWED_IMPORTS = {
    "numpy", "math", "collections", "itertools", "functools", "heapq", "re", "string",
    "copy", "dataclasses", "typing", "enum", "operator", "bisect", "json", "textwrap",
}

_SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in (
        "range", "len", "min", "max", "abs", "enumerate", "zip", "list", "dict",
        "set", "tuple", "int", "float", "bool", "str", "sum", "sorted", "any",
        "all", "map", "filter", "reversed", "round", "divmod", "pow", "isinstance",
        "getattr", "setattr", "hasattr", "delattr", "Exception", "ValueError", "IndexError",
        "KeyError", "TypeError", "ZeroDivisionError", "slice", "frozenset", "bytes",
        "chr", "ord", "repr", "hash", "type", "iter", "next", "callable", "object",
        "StopIteration", "AttributeError", "RuntimeError", "AssertionError", "NotImplementedError",
        "property", "staticmethod", "classmethod", "super", "id", "format", "bytearray",
        "print", "issubclass", "vars", "dir", "globals", "locals", "abs", "hex", "bin",
    ) if hasattr(builtins, name)
}


class _LocalFinder:

    def __init__(self, workdir: Optional[Path], main_ns: dict) -> None:
        self.workdir = Path(workdir).resolve() if workdir else None
        self.main_ns = main_ns
        self.cache: dict = {}
        self.sources: dict = {}

    def local_path(self, name: str) -> Optional[Path]:
        if self.workdir is None or "." in name or not name.isidentifier():
            return None
        p = (self.workdir / f"{name}.py").resolve()
        if p.is_file() and p.parent == self.workdir and p.name != "world_model.py":
            return p
        return None

    def load(self, name: str):
        if name in self.cache:
            return self.cache[name]
        path = self.local_path(name)
        if path is None:
            return None
        import types
        mod = types.ModuleType(name)
        mod.__file__ = str(path)
        mod.__dict__.update({
            "__builtins__": self.main_ns["__builtins__"],
            "np": np, "numpy": np,
            "ENTRY_OBS": self.main_ns.get("ENTRY_OBS"),
            "ENTRY_META": self.main_ns.get("ENTRY_META"),
            "WORKDIR": self.main_ns.get("WORKDIR"),
        })
        self.cache[name] = mod
        src = path.read_text(encoding="utf-8")
        self.sources[name] = hashlib.sha256(src.encode("utf-8")).hexdigest()
        exec(compile(src, str(path), "exec"), mod.__dict__)
        return mod


def _make_import(finder: _LocalFinder):
    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".")[0]
        mod = finder.load(root)
        if mod is not None:
            if not fromlist:
                return mod
            for attr in fromlist:
                if not hasattr(mod, attr):
                    raise ImportError(f"cannot import name '{attr}' from '{root}'")
            return mod
        if root not in _ALLOWED_IMPORTS:
            raise ImportError(
                f"import '{name}' is not allowed — use the standard-library subset "
                f"({', '.join(sorted(_ALLOWED_IMPORTS))}) or your own .py files in the workdir")
        return __import__(name, globals, locals, fromlist, level)
    return _safe_import


def _exec_code(code: str, workdir: Optional[Path] = None) -> dict:
    ns: dict = {
        "np": np, "numpy": np,
        "ENTRY_OBS": None, "ENTRY_META": None,
        "WORKDIR": None if workdir is None else str(workdir),
    }
    finder = _LocalFinder(workdir, ns)
    ns["__builtins__"] = {**_SAFE_BUILTINS, "__import__": _make_import(finder)}
    ns["__name__"] = "world_model"
    exec(compile(code, "<world_model>", "exec"), ns)
    ns["_finder"] = finder
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
    pl, al = pred.splitlines(), actual.splitlines()
    if len(pl) != len(al):
        return (f"{len(pl)} line(s) predicted vs {len(al)} actual "
                f"(first differing line {next((i for i, (a, b) in enumerate(zip(pl, al)) if a != b), min(len(pl), len(al)))})")
    bad = [i for i, (a, b) in enumerate(zip(pl, al)) if a != b]
    cells = sum(sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))
                for a, b in zip(pl, al))
    row = bad[0]
    col = next((i for i, (x, y) in enumerate(zip(pl[row], al[row])) if x != y), 0)
    return (f"{len(bad)} line(s), {cells} character(s) differ; first at line {row} column {col} "
            f"(predicted {pl[row][col:col + 12]!r} vs actual {al[row][col:col + 12]!r})")


def frames_match(pred, actual) -> Optional[str]:
    p, a = _to_text(pred), _to_text(actual)
    return None if p == a else diff_summary(p, a)


def predicted_room(meta: dict, action: str, info: Optional[dict], state=None) -> Optional[str]:
    act = _canonical(action)
    if act.startswith("go to level"):
        parts = act.replace("go to level", "").split()
        return f"level_{parts[0].upper()}x{parts[1].upper()}" if len(parts) >= 2 else None
    src = _norm_room((meta or {}).get("room"))
    for dest in ((info or {}).get("room"), state.get("room") if isinstance(state, dict) else None):
        if not isinstance(dest, str) or not _ROOM_RE.match(dest.strip()):
            continue
        dest = _norm_room(dest)
        if dest != src:
            return dest
    return None


_ROOM_RE = __import__("re").compile(r"^(?:level_)?([A-P])\s*[x×]\s*([A-P])$", __import__("re").I)


@__import__("functools").lru_cache(maxsize=4096)
def _canonical_cached(action: str) -> str:
    text = " ".join(action.strip().split())
    try:
        from env.maze_env import canonical_action
        text = canonical_action(text)
    except Exception:
        pass
    return " ".join(text.lower().split())


def _canonical(action) -> str:
    return _canonical_cached(str(action or ""))


def _norm_room(value) -> str:
    text = str(value or "").strip()
    m = _ROOM_RE.match(text)
    return f"level_{m.group(1).upper()}x{m.group(2).upper()}" if m else text


def advance_meta(meta: dict, action: str, info: Optional[dict] = None, state=None) -> dict:
    out = dict(meta or {})
    info = info or {}
    if info.get("room_changed"):
        dest = predicted_room(meta, action, info, state)
        if dest:
            out["room"] = dest
            seen = list(out.get("visited_rooms") or [])
            if dest not in seen:
                out["visited_rooms"] = seen + [dest]
    act = _canonical(action)
    view = str(out.get("view") or "top-diagonal")
    pitch = VIEW_NAMES.index(view) if view in VIEW_NAMES else 1
    yaw = int(out.get("yaw") or 0) % 4
    if act == "rotate camera up":
        pitch = max(0, pitch - 1)
    elif act == "rotate camera down":
        pitch = min(4, pitch + 1)
    elif act == "rotate camera left":
        yaw = (yaw - 1) % 4
    elif act == "rotate camera right":
        yaw = (yaw + 1) % 4
    elif act.startswith("go to level"):
        parts = act.replace("go to level", "").split()
        if len(parts) >= 2:
            out["room"] = f"level_{parts[0].upper()}x{parts[1].upper()}"
    out["view"], out["yaw"] = VIEW_NAMES[pitch], yaw
    gem = info.get("gem")
    if gem:
        out["gems"] = int(out.get("gems") or 0) + (int(gem) if not isinstance(gem, bool) else 1)
    return out


class CodeWorldModel:

    def __init__(self, code: str, workdir: "str | Path | None" = None) -> None:
        ns = _exec_code(code, Path(workdir) if workdir else None)
        predict = ns.get("predict")
        step = ns.get("step")
        self.stateful = callable(predict)
        if not self.stateful and not callable(step):
            raise ValueError(
                "world-model code must define predict(state, obs, action, meta) "
                "OR step(obs, action, meta)")
        self.code = code
        self.workdir = Path(workdir) if workdir else None
        self._ns = ns
        self._predict = predict if self.stateful else None
        self._step = step if callable(step) else None
        self._predict_arity = _arity(predict) if self.stateful else 0
        self._step_arity = _arity(step) if callable(step) else 0
        init = ns.get("init_state")
        self._init_state = init if callable(init) else None
        self._init_arity = _arity(init) if callable(init) else 0
        # init_state(entry_obs, meta, prev_state): the model carries its own state across segments.
        self.carries = bool(self.stateful and self._init_state is not None and self._init_arity >= 3)

        def _pred(name):
            fn = ns.get(name)
            return (fn, _arity(fn)) if callable(fn) else (None, 0)

        self._bfs_goal = _pred("is_bfs_goal")[0]
        self._heur = _pred("bfs_heuristic")[0]
        self._moves = _pred("bfs_moves")[0]
        self._key = _pred("bfs_state_key")[0]

    def set_entry(self, entry: Optional[dict]) -> None:
        entry = entry or {}
        self._entry = {"obs": entry.get("obs"), "meta": dict(entry.get("meta") or {})}
        obs = entry.get("obs")
        self._ns["ENTRY_OBS"] = None if obs is None else _to_text(obs)
        self._ns["ENTRY_META"] = dict(entry.get("meta") or {}) or None
        for mod in getattr(self._ns.get("_finder"), "cache", {}).values():
            mod.__dict__["ENTRY_OBS"] = self._ns["ENTRY_OBS"]
            mod.__dict__["ENTRY_META"] = self._ns["ENTRY_META"]

    def current_entry(self) -> dict:
        return dict(getattr(self, "_entry", None) or {})

    def init_state(self, entry_obs, meta: Optional[dict] = None, prev_state=None):
        if not self.stateful:
            return None
        if self._init_state is None:
            return {}
        obs = None if entry_obs is None else _to_text(entry_obs)
        if self.carries:
            return self._init_state(obs, dict(meta or {}), prev_state)
        if self._init_arity >= 2:
            return self._init_state(obs, dict(meta or {}))
        return self._init_state(obs)

    def predict(self, state, obs, action, meta: Optional[dict] = None) -> "tuple[str, dict, object]":
        m = dict(meta or {})
        if self.stateful:
            out = (self._predict(state, _to_text(obs), str(action), m)
                   if self._predict_arity >= 4 else
                   self._predict(state, _to_text(obs), str(action)))
            return self._unpack(out, state)
        out = (self._step(_to_text(obs), str(action), m) if self._step_arity >= 3
               else self._step(_to_text(obs), str(action)))
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

    def step(self, obs, action, meta: Optional[dict] = None) -> "tuple[str, dict]":
        st = self.init_state(None, meta) if self.stateful else None
        o, info, _ = self.predict(st, obs, action, meta)
        return o, info

    @property
    def has_bfs_goal(self) -> bool:
        return self._bfs_goal is not None

    def bfs_goal(self, state, obs, info: Optional[dict] = None, args: Optional[dict] = None) -> bool:
        if self._bfs_goal is None:
            return False
        return bool(self._bfs_goal(state, _to_text(obs), dict(info or {}), dict(args or {})))

    @property
    def has_heuristic(self) -> bool:
        return self._heur is not None

    def heuristic(self, obs, state=None, args: Optional[dict] = None) -> float:
        if self._heur is None:
            raise AttributeError("world model defines no bfs_heuristic")
        return float(self._heur(state, _to_text(obs), dict(args or {})))

    @property
    def has_bfs_moves(self) -> bool:
        return self._moves is not None

    def bfs_moves(self, state, obs, args: Optional[dict] = None) -> list:
        if self._moves is None:
            raise AttributeError("world model defines no bfs_moves")
        seqs = []
        for item in (self._moves(state, _to_text(obs), dict(args or {})) or []):
            if isinstance(item, str):
                seqs.append([item])
            elif isinstance(item, (list, tuple)) and all(isinstance(a, str) for a in item):
                seqs.append([str(a) for a in item])
            else:
                raise ValueError("bfs_moves must return a list whose items are action strings or "
                                 f"lists of action strings; got {item!r}")
        return seqs

    @property
    def has_state_key(self) -> bool:
        return self._key is not None

    def state_key(self, state, obs, args: Optional[dict] = None):
        if self._key is None:
            raise AttributeError("world model defines no bfs_state_key")
        k = self._key(state, _to_text(obs), dict(args or {}))
        try:
            hash(k)
        except TypeError:
            k = repr(k)
        return k

    def version(self) -> str:
        h = hashlib.sha256(self.code.encode("utf-8"))
        for name, digest in sorted(getattr(self._ns.get("_finder"), "sources", {}).items()):
            h.update(f"\0{name}\0{digest}".encode("utf-8"))
        return h.hexdigest()[:12]

    def describe(self) -> dict:
        return {"loaded": True, "stateful": self.stateful,
                "has_bfs_goal": self.has_bfs_goal, "has_heuristic": self.has_heuristic,
                "has_bfs_moves": self.has_bfs_moves, "has_state_key": self.has_state_key,
                "local_modules": sorted(getattr(self._ns.get("_finder"), "cache", {})),
                "version": self.version()}


def rollout_state(world: CodeWorldModel, timeline: list, entries: dict, segment: int, end: int,
                  *, start: int = 0, state=None, meta: Optional[dict] = None, start_state=None):
    entry = entries.get(segment) or {}
    world.set_entry(entry)
    if start <= 0:
        state = (copy.deepcopy(start_state) if start_state is not None
                 else world.init_state(entry.get("obs"), entry.get("meta")))
        meta = dict(entry.get("meta") or {})
        start = 0
    else:
        meta = dict(meta or entry.get("meta") or {})
    errors: list = []
    for ts in timeline[start:end]:
        if int(getattr(ts, "segment", 0)) != int(segment) or getattr(ts, "invalid", False):
            continue
        try:
            _, info, state = world.predict(state, ts.before_obs, ts.action, meta)
        except Exception as e:
            errors.append(f"predict() raised {type(e).__name__}: {e} at step {ts.env_step_index}")
            break
        meta = dict(getattr(ts, "after_meta", None) or advance_meta(meta, ts.action, info, state))
    return state, meta, errors


def _checkable(ts) -> bool:
    return not bool(getattr(ts, "invalid", False))


def _flag_mismatches(info: dict, ts) -> "tuple[list, list]":
    errors, kinds = [], []
    for name in FLAG_NAMES:
        got = bool(info.get(name))
        want = bool(getattr(ts, name))
        if got != want:
            errors.append(f"info[{name!r}] predicted {got}, actually {want}")
            kinds.append(f"flag:{name}")
    return errors, kinds


def action_kind(action: str) -> str:
    a = " ".join(str(action or "").strip().lower().split())
    if a in ("up", "down", "left", "right"):
        return "move"
    if a.startswith("rotate camera"):
        return "camera"
    if a == "undo":
        return "undo"
    if a == "reset":
        return "reset"
    if a.startswith("go to level"):
        return "goto"
    return "other"


def _bt_segment(world: CodeWorldModel, timeline: list, entries: dict, seg: int, idxs: list) -> dict:
    return _seg_pass(world, timeline, entries, seg, idxs)[0]


_UNSET = object()


def _seg_pass(world: CodeWorldModel, timeline: list, entries: dict, seg: int, idxs: list,
              start=_UNSET, collect: bool = True, start_error: str = "") -> "tuple[dict, object, dict]":
    """Replay one segment. Without `start` the segment begins from init_state (the classic path);
    with it, from a copy of that state (a carried chain start). Returns the per-transition results,
    the state after the last transition and its meta. collect=False only advances the state."""
    out: dict = {}
    entry = entries.get(seg) or {}
    state = None if start is _UNSET else copy.deepcopy(start)
    meta: dict = dict(entry.get("meta") or {})
    for n, i in enumerate(idxs):
        ts = timeline[i]
        if n == 0:
            world.set_entry(entry)
            meta = dict(entry.get("meta") or getattr(ts, "before_meta", {}) or {})
            if start_error:
                if collect:
                    out[i] = (None, {}, [start_error], ["error"])
                continue
            if start is _UNSET:
                try:
                    state = world.init_state(entry.get("obs"), meta)
                except Exception as e:
                    out[i] = (None, {}, [f"init_state raised {type(e).__name__}: {e}"], ["error"])
                    state = None
                    continue
        world.set_entry(entry)
        errors: list = []
        kinds: list = []
        pred = None
        info: dict = {}
        try:
            pred, info, next_state = world.predict(state, ts.before_obs, ts.action, meta)
        except Exception as e:
            errors.append(f"predict() raised {type(e).__name__}: {e}")
            kinds.append("error")
        else:
            if collect:
                why = frames_match(pred, ts.after)
                if why is not None:
                    errors.append(f"next observation differs: {why}")
                    kinds.append("obs")
                f_err, f_kinds = _flag_mismatches(info, ts)
                errors += f_err
                kinds += f_kinds
            state = next_state
        meta = dict(getattr(ts, "after_meta", None) or advance_meta(meta, ts.action, info, state))
        if collect:
            out[i] = (pred, info, errors, kinds) if errors else (None, {}, errors, kinds)
    return out, state, meta


_BT: dict = {}
_BT_MIN_TRANSITIONS = 300


def _bt_workers() -> int:
    return _bfs_workers()


def _bt_portable(info) -> dict:
    import pickle
    try:
        pickle.dumps(info)
        return info
    except Exception:
        return {k: bool((info or {}).get(k)) for k in FLAG_NAMES}


_BT_SIG_DEPTH = 2
_BT_SIG_ITEMS = 8
_BT_SKIP = frozenset({"__builtins__", "__name__", "__file__", "__doc__", "__spec__", "__loader__",
                      "__package__", "np", "numpy", "ENTRY_OBS", "ENTRY_META", "WORKDIR", "_finder"})
_BT_SERIAL_ONLY: set = set()
_BT_LAST: dict = {}
_BT_OWN_MODULES: set = set()


def _bt_sig(v, depth: int):
    import types
    from itertools import islice
    t = type(v)
    if v is None or t in (bool, int, float, complex):
        return v
    if t in (str, bytes):
        return (len(v), hash(v))
    if isinstance(v, types.ModuleType):
        return ("module", id(v))
    if isinstance(v, np.ndarray):
        return ("nd", id(v), v.shape, v.dtype.str,
                hash(v.tobytes()) if v.nbytes <= (1 << 16) else v.nbytes)
    if callable(getattr(v, "cache_info", None)):
        try:
            return ("cached", id(v), v.cache_info().currsize)
        except Exception:
            return ("cached", id(v))
    if isinstance(v, types.FunctionType):
        return ("fn", id(v), _bt_sig(v.__defaults__, depth), _bt_sig(v.__kwdefaults__, depth))
    if isinstance(v, (types.BuiltinFunctionType, types.MethodType)):
        return ("callable", id(v))
    try:
        size = len(v)
    except Exception:
        size = None
    if depth <= 0:
        return (t.__name__, id(v), size)
    if isinstance(v, dict):
        return ("dict", id(v), size, tuple((_bt_sig(k, 0), _bt_sig(x, depth - 1))
                                          for k, x in islice(v.items(), _BT_SIG_ITEMS)))
    if isinstance(v, (list, tuple, set, frozenset, deque, bytearray)):
        return (t.__name__, id(v), size, tuple(_bt_sig(x, depth - 1)
                                              for x in islice(v, _BT_SIG_ITEMS)))
    if isinstance(v, type) and getattr(v, "__module__", None) not in _BT_OWN_MODULES:
        return ("cls", id(v))
    d = vars(v) if isinstance(v, type) else getattr(v, "__dict__", None)
    if isinstance(d, (dict, types.MappingProxyType)):
        return (t.__name__, id(v), _bt_sig({k: x for k, x in d.items()
                                            if not str(k).startswith("__")}, depth - 1))
    return (t.__name__, id(v), size)


def _bt_fingerprint(world) -> dict:
    spaces = {"<main>": world._ns}
    for name, mod in getattr(world._ns.get("_finder"), "cache", {}).items():
        spaces[name] = mod.__dict__
    _BT_OWN_MODULES.clear()
    _BT_OWN_MODULES.update(n for n in spaces if n != "<main>")
    _BT_OWN_MODULES.add(str(world._ns.get("__name__") or "world_model"))
    out = {name: tuple((k, _bt_sig(v, _BT_SIG_DEPTH)) for k, v in sorted(d.items(), key=lambda kv: str(kv[0]))
                       if k not in _BT_SKIP)
           for name, d in spaces.items()}
    try:
        out["<np.random>"] = hash(np.random.get_state()[1].tobytes())
    except Exception:
        pass
    return out


def _bt_changed(before: dict, after: dict) -> bool:
    return any(after.get(name) != sig for name, sig in before.items())


def _bt_task(segs: list):
    w, tl, en, groups = _BT["world"], _BT["timeline"], _BT["entries"], _BT["groups"]
    before = _BT.get("_fp")
    if before is None:
        before = _bt_fingerprint(w)
    out: dict = {}
    for seg in segs:
        for i, (pred, info, errors, kinds) in _bt_segment(w, tl, en, seg, groups[seg]).items():
            out[i] = (pred, _bt_portable(info) if errors else {}, errors, kinds)
    after = _bt_fingerprint(w)
    _BT["_fp"] = after
    return out, _bt_changed(before, after)


_BT_TASKS_PER_WORKER = 6


def _bt_chunks(groups: dict, n_workers: int) -> list:
    order = sorted(groups, key=lambda s: -len(groups[s]))
    total = sum(len(groups[s]) for s in order)
    target = max(1, total // max(1, n_workers * _BT_TASKS_PER_WORKER))
    chunks, cur, acc = [], [], 0
    for s in order:
        cur.append(s)
        acc += len(groups[s])
        if acc >= target:
            chunks.append(cur)
            cur, acc = [], 0
    if cur:
        chunks.append(cur)
    return chunks


def _bt_parallel(world, timeline, entries, groups: dict, workers: int) -> "dict | str | None":
    import multiprocessing as mp
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        return None
    n = min(workers, len(groups))
    chunks = _bt_chunks(groups, n)
    import pickle
    _BT.update(world=world, timeline=timeline, entries=entries, groups=groups)
    pool = None
    try:
        pool = ctx.Pool(n, initializer=_par_init)
        out: dict = {}
        for part, changed in pool.imap_unordered(_bt_task, chunks, chunksize=1):
            if changed:
                return "mutated"
            out.update(part)
        return out
    except (OSError, EOFError, MemoryError, pickle.PicklingError, mp.ProcessError,
            AttributeError, TypeError, ValueError):
        return None
    finally:
        if pool is not None:
            pool.terminate()
        _BT.clear()


# ── carried state across segments ────────────────────────────────────────────────
# A model whose init_state takes a third parameter gets, at the start of every segment, its own
# state right after the action that ended the previous one (None at segment 0). Segment starts are
# then a chain — start k depends on every transition before it — so they are cached per model
# version in SegmentStarts. When the code changes, the previous version's starts are kept as
# GUESSES: workers replay each segment in parallel from its guessed start and check whether the
# start they hand on equals the next guess. The main process walks the chain in order, takes every
# segment whose start was confirmed, and replays serially only from the first wrong guess until
# the chain agrees with the guesses again. Usually a code change does not touch what is carried,
# so this costs one parallel pass; at worst it is one serial replay.

def _same(a, b) -> bool:
    if a is b:
        return True
    try:
        r = a == b
        if isinstance(r, (bool, np.bool_)):
            return bool(r)
    except Exception:
        pass
    try:
        import pickle
        return pickle.dumps(a, 4) == pickle.dumps(b, 4)
    except Exception:
        return False


class SegmentStarts:
    """Start state of every segment for one model version (only used for carrying models)."""

    def __init__(self) -> None:
        self.version: Optional[str] = None
        self.starts: dict = {}          # seg -> start state under self.version
        self.errors: dict = {}          # seg -> init_state error message
        self.guess: dict = {}           # seg -> start state under an earlier version

    def use(self, world: "CodeWorldModel") -> None:
        ver = world.version()
        if ver != self.version:
            self.guess.update(self.starts)
            self.starts, self.errors, self.version = {}, {}, ver

    def get(self, seg: int):
        return self.starts.get(int(seg), _UNSET)

    def cached(self, world: "CodeWorldModel", seg: int):
        """The start of `seg` if it is already known for world's current code, else _UNSET."""
        return self.get(seg) if self.version == world.version() else _UNSET


def _next_start(world, entries: dict, seg: int, prev_state) -> "tuple[object, str]":
    entry = entries.get(seg) or {}
    world.set_entry(entry)
    try:
        return world.init_state(entry.get("obs"), dict(entry.get("meta") or {}), prev_state), ""
    except Exception as e:
        return None, f"init_state raised {type(e).__name__}: {e}"


def _chain_task(segs: list):
    w, tl, en, groups = _BT["world"], _BT["timeline"], _BT["entries"], _BT["groups"]
    starts, guess, upto, collect = _BT["starts"], _BT["guess"], _BT["upto"], _BT["collect"]
    import pickle
    before = _BT.get("_fp")
    if before is None:
        before = _bt_fingerprint(w)
    rows = []
    for j in segs:
        start = starts.get(j, guess.get(j, _UNSET))
        out, end, _ = _seg_pass(w, tl, en, j, groups.get(j, []), start=start, collect=j in collect)
        out = {i: (pred, _bt_portable(info) if errors else {}, errors, kinds)
               for i, (pred, info, errors, kinds) in out.items()}
        ok_next, blob = None, None
        if j < upto:
            nxt, err = _next_start(w, en, j + 1, end)
            target = starts.get(j + 1, guess.get(j + 1, _UNSET))
            ok_next = (not err and target is not _UNSET and _same(nxt, target))
            if not ok_next and not err:
                try:
                    blob = pickle.dumps(nxt, 4)
                    if len(blob) > (64 << 20):
                        blob = None
                except Exception:
                    blob = None
        rows.append((j, out, ok_next, blob))
    after = _bt_fingerprint(w)
    _BT["_fp"] = after
    return rows, _bt_changed(before, after)


def _chain_parallel(world, timeline, entries, groups, segs, starts, guess, upto, collect, workers):
    import multiprocessing as mp
    import pickle
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        return None
    sizes = {j: groups.get(j, []) for j in segs}
    n = min(workers, len(segs))
    chunks = _bt_chunks(sizes, n)
    _BT.update(world=world, timeline=timeline, entries=entries, groups=groups,
               starts=starts, guess=guess, upto=upto, collect=collect)
    pool = None
    try:
        pool = ctx.Pool(n, initializer=_par_init)
        got: dict = {}
        for rows, changed in pool.imap_unordered(_chain_task, chunks, chunksize=1):
            if changed:
                return "mutated"
            for j, out, ok_next, blob in rows:
                got[j] = (out, ok_next, blob)
        return got
    except (OSError, EOFError, MemoryError, pickle.PicklingError, mp.ProcessError,
            AttributeError, TypeError, ValueError):
        return None
    finally:
        if pool is not None:
            pool.terminate()
        _BT.clear()


def chain_segments(world: "CodeWorldModel", timeline: list, entries: dict, cache: SegmentStarts,
                   upto: int, collect=frozenset(), workers: "int | None" = None) -> dict:
    """Make cache.starts cover segments 0..upto for this model version and return the backtest
    results of the segments in `collect` (computed on the way). Stats go to _BT_LAST."""
    import pickle
    cache.use(world)
    collect = {int(j) for j in collect if int(j) <= upto}
    groups: dict = {}
    for i, ts in enumerate(timeline):
        if _checkable(ts):
            groups.setdefault(int(ts.segment), []).append(i)
    starts, errors, guess = cache.starts, cache.errors, cache.guess
    if 0 not in starts:
        entry = entries.get(0) or {}
        world.set_entry(entry)
        try:
            starts[0] = world.init_state(entry.get("obs"), dict(entry.get("meta") or {}), None)
        except Exception as e:
            starts[0], errors[0] = None, f"init_state raised {type(e).__name__}: {e}"
    first_unknown = next((k for k in range(upto + 1) if k not in starts), upto + 1)
    todo = sorted(collect | set(range(max(0, first_unknown - 1), upto)))
    stats = {"mode": "chain", "segments": len(todo), "serial": 0, "confirmed": 0,
             "starts_computed": max(0, upto + 1 - first_unknown)}
    known_before = set(starts)
    par: dict = {}
    w = _bt_workers() if workers is None else max(1, int(workers))
    runnable = [j for j in todo if j in starts or j in guess]
    ver = cache.version
    if (w > 1 and len(runnable) > 1 and ver not in _BT_SERIAL_ONLY
            and sum(len(groups.get(j, [])) for j in runnable) >= _BT_MIN_TRANSITIONS):
        got = _chain_parallel(world, timeline, entries, groups, runnable, dict(starts), guess,
                              upto, collect, w)
        if got == "mutated":
            _BT_SERIAL_ONLY.add(ver)
            stats["mode"] = "chain-serial-fallback"
        elif got is None:
            stats["mode"] = "chain-serial-poolerror"
        else:
            par = got
            stats["mode"] = "chain-parallel"
    from_guess: set = set()
    raw: dict = {}
    for j in todo:
        start = starts.get(j, _UNSET)
        if start is _UNSET:                       # unreachable: the walk fills starts in order
            break
        used = (j in known_before) or (j in from_guess) or (j in guess and _same(guess[j], start))
        hit = par.get(j) if used and not errors.get(j) else None
        if hit is not None:
            out, ok_next, blob = hit
            raw.update(out)
            if j < upto and (j + 1) not in starts:
                if ok_next:
                    starts[j + 1] = guess[j + 1]
                    from_guess.add(j + 1)
                    stats["confirmed"] += 1
                    continue
                if blob is not None:
                    starts[j + 1] = pickle.loads(blob)
                    continue
            else:
                continue
        stats["serial"] += 1
        out, end, _ = _seg_pass(world, timeline, entries, j, groups.get(j, []), start=start,
                                collect=j in collect, start_error=errors.get(j, ""))
        raw.update(out)
        if j < upto and (j + 1) not in starts:
            starts[j + 1], err = _next_start(world, entries, j + 1, end)
            if err:
                errors[j + 1] = err
    _BT_LAST.clear()
    _BT_LAST.update(stats, transitions=sum(len(groups.get(j, [])) for j in collect))
    return raw


def backtest_rollout(world: CodeWorldModel, timeline: list, entries: dict,
                     segments: "set | None" = None, workers: "int | None" = None,
                     starts: "SegmentStarts | None" = None) -> dict:
    if starts is not None and getattr(world, "carries", False):
        want = {int(ts.segment) for ts in timeline if _checkable(ts)}
        if segments is not None:
            want &= {int(x) for x in segments}
        raw = chain_segments(world, timeline, entries, starts, max(want), collect=want,
                             workers=workers) if want else {}
        return _bt_results(timeline, raw)
    groups: dict = {}
    for i, ts in enumerate(timeline):
        if not _checkable(ts):
            continue
        seg = int(ts.segment)
        if segments is not None and seg not in segments:
            continue
        groups.setdefault(seg, []).append(i)
    w = _bt_workers() if workers is None else max(1, int(workers))
    total = sum(len(v) for v in groups.values())
    raw = None
    mode = "serial"
    if w > 1 and len(groups) > 1 and total >= _BT_MIN_TRANSITIONS:
        try:
            ver = world.version()
        except Exception:
            ver = None
        if ver is not None and ver in _BT_SERIAL_ONLY:
            mode = "serial-known"
        else:
            raw = _bt_parallel(world, timeline, entries, groups, w)
            if raw == "mutated":
                if ver is not None:
                    _BT_SERIAL_ONLY.add(ver)
                raw, mode = None, "serial-fallback"
            elif raw is None:
                mode = "serial-poolerror"
            else:
                mode = "parallel"
    _BT_LAST.clear()
    _BT_LAST.update(mode=mode, transitions=total, segments=len(groups))
    if raw is None:
        raw = {}
        for seg in sorted(groups):
            raw.update(_bt_segment(world, timeline, entries, seg, groups[seg]))
    return _bt_results(timeline, raw)


def _bt_results(timeline: list, raw: dict) -> dict:
    results: dict = {}
    for i in sorted(raw):
        ts = timeline[i]
        pred, info, errors, kinds = raw[i]
        results[i] = {"pred": pred, "info": info, "errors": errors, "kinds": kinds,
                      "terminal": bool(ts.terminal), "before": ts.before_obs, "after": ts.after,
                      "action": ts.action, "kind": action_kind(ts.action)}
    return results


def backtest_summary(results: dict) -> dict:
    by_kind: dict = {}
    ok = bad = 0
    for r in results.values():
        kind = r.get("kind") or "other"
        slot = by_kind.setdefault(kind, {"ok": 0, "bad": 0})
        if r["errors"]:
            bad += 1
            slot["bad"] += 1
        else:
            ok += 1
            slot["ok"] += 1
    total = ok + bad
    return {"ok": ok, "mismatched": bad, "total": total,
            "accuracy": (ok / total) if total else 0.0,
            "by_kind": {k: {**v, "total": v["ok"] + v["bad"],
                            "accuracy": v["ok"] / max(1, v["ok"] + v["bad"])}
                        for k, v in sorted(by_kind.items())}}


_DEFAULT_BFS_ACTIONS = ("up", "down", "left", "right")


def _node_key(obs: str, meta: dict) -> str:
    return f"{meta.get('room','')}|{meta.get('yaw',0)}|{meta.get('view','')}|{hash(obs)}"


_MAIN_SHARE = 0.65
NEAR_K = 5


def _near_add(near: dict, key, h: float, depth: int, path, o, st, m, k: int = NEAR_K) -> None:
    cur = near.get(key)
    if cur is not None and (cur[0], cur[1]) <= (h, depth):
        return
    if cur is None and len(near) >= k:
        worst = max(near, key=lambda q: (near[q][0], near[q][1]))
        if (near[worst][0], near[worst][1]) <= (h, depth):
            return
        del near[worst]
    near[key] = (h, depth, list(path), o, st, dict(m or {}))


def _near_list(near: dict) -> list:
    return [{"h": v[0], "depth": v[1], "path": v[2], "obs": v[3], "state": v[4], "meta": v[5], "key": q}
            for q, v in sorted(near.items(), key=lambda kv: (kv[1][0], kv[1][1]))]
_MIN_FALLBACK_S = 5.0


class _Novelty:

    def __init__(self, obs: str) -> None:
        self.seen = set(enumerate(obs))

    def novel(self, obs: str) -> bool:
        new = set(enumerate(obs)) - self.seen
        if not new:
            return False
        self.seen |= new
        return True


def _result(**kw) -> dict:
    r = {
        "plan": None, "goal_reason": None, "final_obs": None,
        "expanded": 0, "distinct": 0, "frontier": 0,
        "termination": "exhausted", "note": "", "errors": [],
        "algo": "none", "optimal": True, "best": None, "depth_reached": 0,
        "h_start": None, "heuristic_error": None, "predict_ms": None, "secs": 0.0,
        "phases": [], "time_budget_s": None, "has_heuristic": False, "macro": False,
        "custom_key": False, "checkpoint": None, "resumed": False,
        "cross_rooms": False, "workers": 1, "dropped": 0, "near": [],
        "memory": None,
    }
    r.update(kw)
    return r


_PAR: dict = {}
_POOL: dict = {"pool": None, "world": None, "version": None, "workers": 0}
_LOCAL: dict = {"sid": None, "seen": {}, "ctx": None, "ctx_key": None}
_SID = [0]
_PAR_START_S = 0.3
_PAR_OVERSUB = 3


def _bfs_workers() -> int:
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 1
    return max(1, min(n - 1, 63))


def _par_min_predict_ms() -> float:
    return 0.05


def _par_start_s() -> float:
    return _PAR_START_S


_MEM_FRACTION = 0.6
_MEM_CHECK_S = 1.0


def _mem_fraction() -> float:
    return _MEM_FRACTION


def _read_int(p: Path) -> Optional[int]:
    try:
        raw = p.read_text().strip()
    except OSError:
        return None
    return int(raw) if raw.isdigit() else None


def _stat_anon(p: Path) -> Optional[int]:
    try:
        vals = dict(line.split()[:2] for line in p.read_text().splitlines() if line.strip())
    except (OSError, ValueError):
        return None
    if "anon" not in vals:
        return None
    return int(vals["anon"]) + int(vals.get("shmem", 0))


def memory_usage(proc_cgroup: str = "/proc/self/cgroup", cgroup_root: str = "/sys/fs/cgroup",
                 meminfo: str = "/proc/meminfo") -> Optional[tuple]:
    try:
        rel = next((ln.split(":", 2)[2] for ln in Path(proc_cgroup).read_text().splitlines()
                    if ln.startswith("0::")), None)
    except OSError:
        rel = None
    if rel is not None:
        root = Path(cgroup_root)
        d = root / rel.strip().lstrip("/")
        best = None
        while True:
            lim = _read_int(d / "memory.max")
            if lim is not None and (best is None or lim <= best[0]):
                best = (lim, d)
            if d == root or root not in d.parents:
                break
            d = d.parent
        if best is not None:
            used = _stat_anon(best[1] / "memory.stat")
            if used is not None:
                return used, best[0]
    try:
        info = dict(line.split(":", 1) for line in Path(meminfo).read_text().splitlines() if ":" in line)
        total = int(info["MemTotal"].split()[0]) * 1024
        avail = int(info["MemAvailable"].split()[0]) * 1024
    except (OSError, KeyError, ValueError, IndexError):
        return None
    return total - avail, total


CHECKPOINT_MEM_FRACTION = 0.3


def memory_heavy() -> bool:
    m = memory_usage()
    return m is not None and m[1] > 0 and m[0] >= CHECKPOINT_MEM_FRACTION * m[1]


def release_memory(close_pool: bool = False) -> None:
    import gc
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass
    if close_pool:
        _close_pool()


def _memory_over() -> Optional[tuple]:
    m = memory_usage()
    if m is None or m[1] <= 0:
        return None
    return m if m[0] >= _mem_fraction() * m[1] else None


_PACK_TAG = "\x00arc-packed-siblings"


class _Packed(bytes):
    __slots__ = ()


def _pack_mode(mode) -> bool:
    return mode not in (False, "0")


def _unpack(st):
    if type(st) is tuple and len(st) == 3 and st[0] == _PACK_TAG:
        import pickle
        return pickle.loads(st[1])[st[2]]
    return st


def _par_init() -> None:
    import sys
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")


def _make_reached(world, args: dict):
    def goal(st, o, info: dict) -> bool:
        try:
            return world.bfs_goal(st, o, info, args)
        except Exception as e:
            raise _GoalError(f"is_bfs_goal raised {type(e).__name__}: {e}") from e

    def reached(info: dict, no: str, nst, m: dict) -> Optional[str]:
        if info.get("won"):
            return "won"
        return "is_goal" if goal(nst, no, info) else None
    return goal, reached


def _hooks(world, args: dict, *, use_h: bool, use_moves: bool, use_key: bool):
    h_fn = (lambda o, st: world.heuristic(o, st, args)) if use_h else None
    moves_fn = (lambda st, o, m: world.bfs_moves(st, o, args)) if use_moves else None
    key_fn = (lambda st, o: world.state_key(st, o, args)) if use_key else None
    return h_fn, moves_fn, key_fn


def _worker_ctx(spec: dict) -> dict:
    entry = spec.get("entry")
    ck = (spec["sid"], spec["h_dead"], None if entry is None else hash((str(entry.get("obs")), repr(sorted((entry.get("meta") or {}).items())))))
    if _LOCAL["ctx_key"] == ck:
        return _LOCAL["ctx"]
    world = _PAR["world"]
    if hasattr(world, "set_entry") and spec.get("entry") is not None:
        world.set_entry(spec["entry"])
    _, reached = _make_reached(world, spec["args"])
    h_fn, moves_fn, key_fn = _hooks(world, spec["args"], use_h=spec["use_h"],
                                    use_moves=spec["use_moves"], use_key=spec["use_key"])
    ctx = {"world": world, "moves": spec["moves"], "moves_fn": moves_fn, "key_fn": key_fn,
           "h_fn": h_fn, "reached": reached, "max_depth": spec["max_depth"],
           "cross_rooms": spec["cross_rooms"], "h_dead": spec["h_dead"]}
    _LOCAL["ctx"], _LOCAL["ctx_key"] = ctx, ck
    return ctx


def _par_task(item):
    spec, (cur, st, m, path) = item
    if _LOCAL["sid"] != spec["sid"]:
        _LOCAL["sid"], _LOCAL["seen"] = spec["sid"], {}
    children, stats = _expand(_worker_ctx(spec), cur, _unpack(st), m, path)
    seen, out, states = _LOCAL["seen"], [], []
    for o, nst, nm, p, why, key, h in children:
        if why is not None:
            out.append([o, None, nm, p, why, None, 0.0])
            continue
        g = len(p)
        prev = seen.get(key)
        if prev is not None and prev <= g:
            stats["dropped"] += 1
            continue
        seen[key] = g
        out.append([o, nst, nm, p, None, key, h])
        states.append(len(out) - 1)
    if spec.get("pack", True) and states:
        import pickle
        blob = _Packed(pickle.dumps([out[i][1] for i in states], protocol=pickle.HIGHEST_PROTOCOL))
        for j, i in enumerate(states):
            out[i][1] = (_PACK_TAG, blob, j)
    return [tuple(r) for r in out], stats


def _close_pool() -> None:
    pool = _POOL.get("pool")
    if pool is not None:
        try:
            pool.terminate()
            pool.join()
        except Exception:
            pass
    _POOL.update(pool=None, world=None, version=None, workers=0)


def _get_pool(world, workers: int):
    try:
        ver = world.version() if hasattr(world, "version") else None
    except Exception:
        ver = None
    if (_POOL["pool"] is not None and _POOL["world"] is world and _POOL["version"] == ver
            and _POOL["workers"] == workers):
        return _POOL["pool"]
    _close_pool()
    try:
        import multiprocessing
        _PAR["world"] = world
        pool = multiprocessing.get_context("fork").Pool(workers, initializer=_par_init)
    except Exception:
        return None
    _POOL.update(pool=pool, world=world, version=ver, workers=workers)
    return pool


def _err(errors: list, msg: str) -> None:
    if msg not in errors and len(errors) < 5:
        errors.append(msg)


def _h(ctx: dict, stats: dict, o, st) -> float:
    h_fn = ctx["h_fn"]
    if h_fn is None or ctx.get("h_dead") or stats["h_err"] is not None:
        return 0.0
    try:
        v = float(h_fn(o, st))
        if not math.isfinite(v):
            raise ValueError(f"returned non-finite value {v!r}")
        return v if v > 0.0 else 0.0
    except Exception as e:
        stats["h_err"] = f"{type(e).__name__}: {e}"
        return 0.0


def _k(ctx: dict, stats: dict, o, st, m):
    key_fn = ctx["key_fn"]
    if key_fn is None:
        return _node_key(o, m)
    try:
        return (m.get("room", ""), m.get("yaw", 0), m.get("view", ""), key_fn(st, o))
    except Exception as e:
        _err(stats["errors"], f"bfs_state_key raised {type(e).__name__}: {e} (used the whole board instead)")
        return _node_key(o, m)


def _new_stats() -> dict:
    return {"calls": 0, "n_pred": 0, "t_pred": 0.0, "errors": [], "h_err": None,
            "hit_depth": False, "dropped": 0}


def _expand(ctx: dict, cur, st, m, path):
    world, moves_fn, reached = ctx["world"], ctx["moves_fn"], ctx["reached"]
    max_depth, cross_rooms = ctx["max_depth"], ctx["cross_rooms"]
    stats = _new_stats()
    macros = [[a] for a in ctx["moves"]]
    if moves_fn is not None:
        try:
            macros = moves_fn(st, cur, m)
        except Exception as e:
            _err(stats["errors"], f"bfs_moves raised {type(e).__name__}: {e} (fell back to `actions`)")
            macros = [[a] for a in ctx["moves"]]
    children: list = []
    for macro in macros:
        if not macro:
            continue
        if len(path) + len(macro) > max_depth:
            stats["hit_depth"] = True
            continue
        o, s2, mm, p = cur, st, m, path
        for act in macro:
            stats["calls"] += 1
            t0 = time.perf_counter()
            try:
                nxt, info, nst = world.predict(s2, o, act, mm)
            except Exception as e:
                stats["t_pred"] += time.perf_counter() - t0
                _err(stats["errors"], f"predict({act!r}) raised {type(e).__name__}: {e}")
                p = None
                break
            stats["t_pred"] += time.perf_counter() - t0
            stats["n_pred"] += 1
            if info.get("dead"):
                p = None
                break
            nmeta = advance_meta(mm, act, info, nst)
            p = p + [act]
            why = reached(info, nxt, nst, nmeta)
            if why is not None:
                children.append((nxt, nst, nmeta, p, why, None, 0.0))
                p = None
                break
            if info.get("room_changed") and not cross_rooms:
                p = None
                break
            o, s2, mm = nxt, nst, nmeta
        if p is not None:
            children.append((o, s2, mm, p, None, _k(ctx, stats, o, s2, mm), _h(ctx, stats, o, s2)))
    return children, stats


def _picklable(obj) -> bool:
    try:
        import pickle
        pickle.dumps(obj)
        return True
    except Exception:
        return False


def _search(world, start, start_state, start_meta, moves, reached, *,
            max_depth, max_nodes, deadline, mode="bfs", h_fn=None, novelty=False,
            moves_fn=None, key_fn=None, resume=None, keep_checkpoint=False,
            cross_rooms=False, workers=1, spec=None) -> dict:
    t_start = time.monotonic()
    ctx = {"world": world, "moves": list(moves), "moves_fn": moves_fn, "key_fn": key_fn,
           "h_fn": h_fn, "reached": reached, "max_depth": max_depth, "cross_rooms": cross_rooms}
    tot = _new_stats()
    errors = tot["errors"]
    t_expand = 0.0

    def merge(s: dict) -> None:
        tot["n_pred"] += s["n_pred"]
        tot["t_pred"] += s["t_pred"]
        tot["dropped"] += s.get("dropped", 0)
        if s["h_err"] is not None and tot["h_err"] is None:
            tot["h_err"] = s["h_err"]
            ctx["h_dead"] = True
        for e in s["errors"]:
            _err(errors, e)

    def prio(g_cost: int, h: float):
        if mode == "astar":
            return (g_cost + h, h)
        if mode == "greedy":
            return (h, g_cost)
        return (g_cost,)

    if resume is not None:
        h0 = resume["h0"]
        best = resume["best"]
        best_g: dict = resume["best_g"]
        heap: list = resume["heap"]
        seq = resume["seq"]
        depth_reached = resume["depth_reached"]
        sid = resume.get("sid") or 0
        near: dict = dict(resume.get("near") or {})
    else:
        h0 = _h(ctx, tot, start, start_state)
        best = (h0, 0, [], start) if h_fn is not None else None
        k0 = _k(ctx, tot, start, start_state, start_meta)
        near = {}
        if h_fn is not None:
            _near_add(near, k0, h0, 0, [], start, start_state, start_meta)
        best_g = {k0: 0}
        heap = []
        seq = 0
        heapq.heappush(heap, (prio(0, h0), seq, start, start_state, start_meta, [], None, k0))
        seq += 1
        depth_reached = 0
        _SID[0] += 1
        sid = _SID[0]
    if tot["h_err"] is not None:
        ctx["h_dead"] = True
    nov = _Novelty(start) if novelty else None
    expanded = 0
    popped = 0
    hit_depth = False
    timed_out = False
    mem_out = None
    next_mem_check = t_start + _MEM_CHECK_S
    algo = mode + ("+novelty" if novelty else "")
    pool = None
    used_pool = False
    pack = False
    par_ok = workers > 1 and spec is not None and not novelty

    def done(plan, reason, final, term, optimal=True) -> dict:
        n = tot["n_pred"]
        ms = ((1000.0 * t_expand / n) if used_pool else (1000.0 * tot["t_pred"] / n)) if n else None
        return _result(
            plan=plan, goal_reason=reason, final_obs=final,
            expanded=expanded, distinct=len(best_g), frontier=len(heap),
            termination=term, errors=list(errors), algo=algo, optimal=bool(optimal),
            best=None if best is None else
                 {"h": best[0], "depth": best[1], "path": list(best[2]), "obs": best[3]},
            near=_near_list(near) if plan is None else [],
            depth_reached=depth_reached, h_start=h0 if h_fn is not None else None,
            heuristic_error=tot["h_err"], predict_ms=ms,
            secs=time.monotonic() - t_start,
            resumed=resume is not None, cross_rooms=bool(cross_rooms),
            workers=workers if used_pool else 1, dropped=tot["dropped"],
            checkpoint=({"heap": heap, "best_g": best_g, "seq": seq, "best": best,
                         "depth_reached": depth_reached, "h0": h0, "mode": mode, "sid": sid,
                         "near": dict(near)}
                        if keep_checkpoint and plan is None and not novelty
                        and term in ("node_cap", "timeout") else None),
            memory=mem_out,
        )

    def push(o, st, m, path, reason, key, h):
        nonlocal seq, depth_reached, best
        if reason is not None:
            if mode != "astar":
                return done(path, reason, o, "found")
            heapq.heappush(heap, (prio(len(path), 0.0), seq, o, st, m, path, reason, None))
            seq += 1
            return None
        prev = best_g.get(key)
        if prev is not None and prev <= len(path):
            return None
        if nov is not None and not nov.novel(o):
            return None
        best_g[key] = len(path)
        if best is not None and (h, len(path)) < (best[0], best[1]):
            best = (h, len(path), path, o)
        if best is not None:
            _near_add(near, key, h, len(path), path, o, st, m)
        if len(path) > depth_reached:
            depth_reached = len(path)
        heapq.heappush(heap, (prio(len(path), h), seq, o, st, m, path, None, key))
        seq += 1
        return None

    def absorb(children, s):
        nonlocal expanded, hit_depth
        expanded += s["calls"]
        hit_depth = hit_depth or s["hit_depth"]
        merge(s)
        for ch in children:
            r = push(*ch)
            if r is not None:
                return r
        return None

    try:
        while heap and expanded < max_nodes:
            now = time.monotonic()
            if now >= deadline:
                timed_out = True
                break
            if now >= next_mem_check:
                next_mem_check = now + _MEM_CHECK_S
                mem_out = _memory_over()
                if mem_out is not None:
                    break
            if (par_ok and pool is None and now - t_start >= _par_start_s() and tot["n_pred"]
                    and 1000.0 * tot["t_pred"] / tot["n_pred"] >= _par_min_predict_ms()):
                par_ok = False
                sample = heap[0][3]
                if _picklable(sample):
                    pool = _get_pool(world, workers)
                    used_pool = pool is not None
                    pack = _pack_mode(spec.get("pack"))
            cap = 1
            if pool is not None:
                per_node = max(1.0, expanded / max(1, popped))
                cap = max(1, min(workers * _PAR_OVERSUB, int((max_nodes - expanded) / per_node) + 1))
            batch: list = []
            while heap and len(batch) < cap:
                e = heap[0]
                if e[6] is not None:
                    if batch:
                        break
                    heapq.heappop(heap)
                    return done(e[5], e[6], e[2], "found")
                heapq.heappop(heap)
                _, _, cur, st, m, path, _, key = e
                if mode == "astar" and best_g.get(key, len(path)) < len(path):
                    continue
                if len(path) >= max_depth:
                    hit_depth = True
                    continue
                batch.append((cur, st, m, path))
            if not batch:
                continue
            popped += len(batch)
            t0 = time.monotonic()
            todo = batch
            if pool is not None:
                ts = dict(spec, sid=sid, h_dead=bool(ctx.get("h_dead")), pack=pack)
                n_done = 0
                try:
                    for children, s in pool.imap(_par_task, [(ts, b) for b in batch], chunksize=1):
                        r = absorb(children, s)
                        n_done += 1
                        if r is not None:
                            t_expand += time.monotonic() - t0
                            return r
                    todo = []
                except _GoalError:
                    raise
                except Exception:
                    _close_pool()
                    pool = None
                    todo = batch[n_done:]
            for cur, st, m, path in todo:
                r = absorb(*_expand(ctx, cur, _unpack(st), m, path))
                if r is not None:
                    t_expand += time.monotonic() - t0
                    return r
            t_expand += time.monotonic() - t0
    except BaseException:
        if pool is not None:
            _close_pool()
        raise

    goals = [e for e in heap if e[6] is not None]
    if goals:
        e = min(goals, key=lambda e: (len(e[5]), e[1]))
        return done(e[5], e[6], e[2], "found", optimal=False)
    term = ("memory_cap" if mem_out is not None else
            "timeout" if timed_out else
            "node_cap" if expanded >= max_nodes else
            "depth_cap" if hit_depth else "exhausted")
    return done(None, None, None, term)


def _phase(r: dict) -> dict:
    return {"algo": r["algo"], "termination": r["termination"], "expanded": r["expanded"],
            "distinct": r["distinct"], "secs": round(float(r["secs"]), 1)}


class _GoalError(Exception):
    pass


def bfs(
    world: CodeWorldModel,
    obs: str,
    meta: dict,
    *,
    actions: "list | tuple" = _DEFAULT_BFS_ACTIONS,
    goal_args: Optional[dict] = None,
    start_state=None,
    max_depth: int = 500,
    max_nodes: int = 5_000_000,
    time_budget_s: float = 1200.0,
    resume: Optional[dict] = None,
    keep_checkpoint: bool = False,
    cross_rooms: bool = False,
    workers: Optional[int] = None,
) -> dict:
    args = dict(goal_args or {})
    use = dict(use_h=bool(getattr(world, "has_heuristic", False)),
               use_moves=bool(getattr(world, "has_bfs_moves", False)),
               use_key=bool(getattr(world, "has_state_key", False)))
    h_fn, moves_fn, key_fn = _hooks(world, args, **use)
    base = dict(time_budget_s=time_budget_s, has_heuristic=h_fn is not None,
                macro=moves_fn is not None, custom_key=key_fn is not None)
    if not world.has_bfs_goal:
        return _result(termination="no_goal_capability", **base,
                       note="world_model.py defines no is_bfs_goal(state, obs, info, args).")

    moves = [str(a) for a in actions]
    start = _to_text(obs)
    start_meta = dict(meta or {})
    goal, reached = _make_reached(world, args)
    n_workers = _bfs_workers() if workers is None else max(1, int(workers))
    spec = dict(args=args, moves=list(moves), max_depth=max_depth, cross_rooms=bool(cross_rooms),
                pack="1",
                entry=world.current_entry() if hasattr(world, "current_entry") else None, **use)
    if n_workers > 1 and not _picklable(spec):
        n_workers = 1

    try:
        if goal(start_state, start, {}):
            return _result(plan=[], goal_reason="is_goal", final_obs=start, distinct=1,
                           termination="found", **base,
                           note="the starting board already satisfies your is_bfs_goal.")
        out = _portfolio(world, start, start_state, start_meta, moves, reached, h_fn,
                         max_depth=max_depth, max_nodes=max_nodes, time_budget_s=time_budget_s,
                         moves_fn=moves_fn, key_fn=key_fn, resume=resume,
                         keep_checkpoint=keep_checkpoint, cross_rooms=cross_rooms,
                         workers=n_workers, spec=spec)
    except _GoalError as e:
        return _result(termination="goal_error", errors=[str(e)], **base)
    return out


def _portfolio(world, start, start_state, start_meta, moves, reached, h_fn, *,
               max_depth: int, max_nodes: int, time_budget_s: float, moves_fn=None,
               key_fn=None, resume=None, keep_checkpoint=False, cross_rooms=False,
               workers=1, spec=None) -> dict:
    t0 = time.monotonic()
    common = dict(max_depth=max_depth, max_nodes=max_nodes, h_fn=h_fn, moves_fn=moves_fn,
                  key_fn=key_fn, cross_rooms=cross_rooms, workers=workers, spec=spec)
    share = (_MAIN_SHARE if h_fn is not None
             and time_budget_s * (1.0 - _MAIN_SHARE) >= _MIN_FALLBACK_S else 1.0)
    r = _search(world, start, start_state, start_meta, moves, reached,
                mode="astar" if h_fn is not None else "bfs",
                deadline=t0 + time_budget_s * share, resume=resume,
                keep_checkpoint=keep_checkpoint, **common)
    phases = [_phase(r)]
    out = r
    if h_fn is not None and r["plan"] is None and r["termination"] in ("node_cap", "timeout", "memory_cap"):
        remaining = t0 + time_budget_s - time.monotonic()
        if remaining >= _MIN_FALLBACK_S:
            if r.get("checkpoint") is not None and memory_heavy():
                r["checkpoint"] = None
                release_memory()
            elif r["termination"] == "memory_cap":
                release_memory()
            r2 = _search(world, start, start_state, start_meta, moves, reached,
                         mode="greedy", deadline=t0 + time_budget_s, **common)
            phases.append(_phase(r2))
            if r2["plan"] is not None:
                out = r2
                out["optimal"] = False
            else:
                merged: dict = {}
                for e in (r.get("near") or []) + (r2.get("near") or []):
                    _near_add(merged, e["key"], e["h"], e["depth"], e["path"], e["obs"], e["state"], e["meta"])
                r["near"] = _near_list(merged)
                b1, b2 = r["best"], r2["best"]
                if b1 is not None and b2 is not None and (b2["h"], b2["depth"]) < (b1["h"], b1["depth"]):
                    r["best"] = b2
                r["depth_reached"] = max(r["depth_reached"], r2["depth_reached"])
                r["heuristic_error"] = r["heuristic_error"] or r2["heuristic_error"]
                if r["predict_ms"] is None:
                    r["predict_ms"] = r2["predict_ms"]
            out["expanded"] = r["expanded"] + r2["expanded"]
            out["errors"] = (r["errors"] + [e for e in r2["errors"] if e not in r["errors"]])[:5]
    out["phases"] = phases
    out["secs"] = time.monotonic() - t0
    out["time_budget_s"] = time_budget_s
    out["has_heuristic"] = h_fn is not None
    out["macro"] = moves_fn is not None
    out["custom_key"] = key_fn is not None
    return out


def tried_actions(world: CodeWorldModel, timeline: list, entries: dict, key_fn, rooms,
                  actions, *, time_budget_s: float = 20.0, start_of=None):
    rooms, actions = set(rooms), set(actions)
    out: set = set()
    groups: dict = {}
    for i, ts in enumerate(timeline):
        if getattr(ts, "invalid", False):
            continue
        groups.setdefault(int(getattr(ts, "segment", 0)), []).append(i)
    want = [seg for seg, idx in groups.items()
            if any((getattr(timeline[i], "before_meta", {}) or {}).get("room") in rooms
                   and timeline[i].action in actions for i in idx)]
    deadline = time.monotonic() + time_budget_s
    ctx = {"key_fn": key_fn}
    stats = _new_stats()
    complete = True
    saved_entry = world.current_entry() if hasattr(world, "current_entry") else None
    try:
        _replay_tried(world, timeline, entries, groups, want, rooms, actions, ctx, stats, out, deadline,
                      start_of=start_of)
    except _HistoryIncomplete:
        complete = False
    finally:
        if saved_entry is not None and hasattr(world, "set_entry"):
            world.set_entry(saved_entry)
    return out, complete


class _HistoryIncomplete(Exception):
    pass


def _replay_tried(world, timeline, entries, groups, want, rooms, actions, ctx, stats, out, deadline,
                  start_of=None):
    incomplete = False
    for seg in sorted(want):
        entry = entries.get(seg) or {}
        state, meta = None, {}
        carried = _UNSET
        if start_of is not None:
            carried = start_of(seg)
            if carried is _UNSET:              # chain not computed for this version yet: skip it
                incomplete = True
                continue
        for n, i in enumerate(groups[seg]):
            if time.monotonic() > deadline:
                raise _HistoryIncomplete()
            ts = timeline[i]
            try:
                if hasattr(world, "set_entry"):
                    world.set_entry(entry)
                if n == 0:
                    meta = dict(entry.get("meta") or getattr(ts, "before_meta", {}) or {})
                    state = (world.init_state(entry.get("obs"), meta) if carried is _UNSET
                             else copy.deepcopy(carried))
                bm = dict(getattr(ts, "before_meta", None) or meta)
                if bm.get("room") in rooms and ts.action in actions:
                    out.add((_k(ctx, stats, ts.before_obs, state, bm), ts.action))
                _, info, state = world.predict(state, ts.before_obs, ts.action, meta)
            except Exception:
                incomplete = True
                break
            meta = dict(getattr(ts, "after_meta", None) or advance_meta(meta, ts.action, info, state))
    if incomplete:
        raise _HistoryIncomplete()


def rollout_actions(world: CodeWorldModel, obs: str, state, meta: dict, actions: list):
    o, st, m = _to_text(obs), state, dict(meta or {})
    for i, a in enumerate(actions, 1):
        o, info, st = world.predict(st, o, a, m)
        if info.get("dead"):
            return None, None, None, f"your model predicts the player dies at prefix action {i} ({a!r})"
        if info.get("room_changed"):
            return None, None, None, (f"your model predicts the room changes at prefix action {i} "
                                      f"({a!r}); a prefix must stay in the current room")
        m = advance_meta(m, a, info, st)
    return o, st, m, ""
