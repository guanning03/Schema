from __future__ import annotations

import copy
import json
import math
import os
import re
import shutil
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

from .timestep import FLAG_NAMES, meta_of
from .world import action_kind, backtest_summary

MODEL_FILE = "world_model.py"
NOTES_FILE = "notes.md"
_PROTECTED = ("events.jsonl", "run.json", "scorecard.json")
_READONLY_DIRS = ("snapshots", "sessions")
_TOOL_BUDGET_S = 180.0
_BFS_DEFAULT_TIME_S = 1200.0
_LONG_TOOL_MAX_S = 1500.0
_BFS_TOOL_BUDGET_S = 1560.0
_BFS_TAIL_S = 45.0
_EXEC_DEFAULT_TIMEOUT_S = 1200.0
_CALL_WINDOW_S = 1620.0
_MIN_CALL_ROOM_S = 10.0
_RUN_GRACE_S = 3600.0
_BFS_DEFAULT_DEPTH = 500
_BFS_DEFAULT_NODES = 5_000_000
ROOM_NOTES_DIR = "rooms"
_ROOM_NOTES_SEED = ("# {room}\n\nYour notes for this room. The harness created this file the first time "
                    "you entered {room}.\n")
_BOARD_CAP = 8
_HISTORY_BYTES = 60_000
BFS_PLAN_FILE = "plans/last_bfs.json"
AT_FILE_HELP = (' An item "@<file>" stands for the action list saved in that JSON file in your '
                'workdir (a list, or {"actions": [...]}).')
CURRENT_BOARD_FILE = "current_board.py"
CURRENT_STATE_FILE = "current_state.pkl"
HISTORY_FILE = "history.jsonl"


def _cam(meta: dict) -> dict:
    return {"room": meta.get("room"), "view": meta.get("view"), "yaw": meta.get("yaw")}


def _harness_dir(workdir: Path, rel: str) -> Optional[Path]:
    p = Path(workdir)
    for part in Path(rel).parts:
        p = p / part
        if p.is_symlink():
            return None
        if not p.exists():
            try:
                os.mkdir(p)
            except OSError:
                return None
        elif not p.is_dir():
            return None
    return p


def _open_regular_for_write(f: Path, flags: int) -> int:
    import stat
    try:
        st = os.lstat(f)
        if not stat.S_ISREG(st.st_mode):
            if stat.S_ISDIR(st.st_mode):
                raise OSError(f"{f} is a directory")
            os.unlink(f)
    except FileNotFoundError:
        pass
    fd = os.open(f, flags | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0), 0o644)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError(f"{f} is not a regular file")
        os.set_blocking(fd, True)
    except BaseException:
        os.close(fd)
        raise
    return fd


def harness_write_json(workdir: Path, rel: str, obj) -> bool:
    d = _harness_dir(workdir, str(Path(rel).parent)) if str(Path(rel).parent) not in ("", ".") else Path(workdir)
    if d is None:
        return False
    f = d / Path(rel).name
    try:
        fd = _open_regular_for_write(f, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(obj))
        return True
    except OSError:
        return False


def harness_clear_dir(workdir: Path, rel: str, pattern: str = "*.json") -> None:
    p = Path(workdir)
    for part in Path(rel).parts:
        p = p / part
        if p.is_symlink() or not p.is_dir():
            return
    for f in p.glob(pattern):
        try:
            if f.is_symlink() or not f.is_dir():
                f.unlink()
        except OSError:
            pass


def clear_stale_plans(workdir: Path) -> None:
    import stat
    harness_clear_dir(workdir, "plans/closest")
    d = Path(workdir) / "plans"
    if d.is_symlink() or not d.is_dir():
        return
    f = d / Path(BFS_PLAN_FILE).name
    try:
        if not stat.S_ISDIR(os.lstat(f).st_mode):
            os.unlink(f)
    except OSError:
        pass


def history_start_record(state) -> dict:
    return {"i": -1, "turn": 0, "segment": 0, "action": None, "refused": False,
            "before": None, "after": _cam(meta_of(state)), "gems": int(state.gems),
            "rooms_visited": len(state.visited_rooms), "gem": False, "room_changed": False,
            "dead": False, "won": False, "transition": state.transition or None,
            "board": state.observation}


def history_record(i: int, turn: int, ts) -> dict:
    st = ts.state
    return {"i": int(i), "turn": int(turn), "segment": int(ts.segment), "action": ts.action,
            "refused": bool(ts.invalid), "before": _cam(ts.before_meta), "after": _cam(ts.after_meta),
            "gems": int(st.gems), "rooms_visited": len(st.visited_rooms), "gem": bool(ts.gem),
            "room_changed": bool(ts.room_changed), "dead": bool(ts.dead), "won": bool(ts.won),
            "transition": st.transition or None, "board": st.observation}


def write_history(workdir: Path, records: list, *, append: bool) -> None:
    p = Path(workdir) / HISTORY_FILE
    fd = _open_regular_for_write(p, os.O_WRONLY | os.O_CREAT | (os.O_APPEND if append else os.O_TRUNC))
    with os.fdopen(fd, "a" if append else "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_current_board(workdir: Path, obs: Optional[str], meta: dict,
                        model_version: Optional[str] = None) -> None:
    p = Path(workdir) / CURRENT_BOARD_FILE
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text("# Written by the harness at the start of every turn; read-only.\n"
                   f"BOARD = {str(obs or '')!r}\nMETA = {dict(meta or {})!r}\n"
                   f"MODEL_VERSION = {model_version!r}\n", encoding="utf-8")
    os.replace(tmp, p)


def room_notes_path(workdir: Path, room: str) -> Path:
    return Path(workdir) / ROOM_NOTES_DIR / f"{room}.md"


def ensure_room_notes(workdir: Path, room: str) -> bool:
    if not room:
        return False
    p = room_notes_path(workdir, room)
    if p.exists():
        return False
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_ROOM_NOTES_SEED.format(room=room), encoding="utf-8")
    return True


def load_plan_file(workdir: Path, raw: Any) -> "tuple[Optional[list], str]":
    text = str(raw or "").strip()
    if not text:
        return None, "ERROR: '@' must be followed by a file name"
    wd = Path(workdir).resolve()
    p = Path(text)
    p = (p if p.is_absolute() else wd / p).resolve()
    if not p.is_relative_to(wd) or p.name in _PROTECTED or not p.is_file():
        return None, f"ERROR: cannot read {text!r} (it must be a file in your workdir)"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except ValueError as e:
        return None, f"ERROR: {text!r} is not valid JSON ({e})"
    acts = data.get("actions") if isinstance(data, dict) else data
    if not isinstance(acts, list) or not acts or not all(isinstance(a, str) for a in acts):
        return None, (f"ERROR: {text!r} must hold a non-empty list of action strings "
                      "(or an object with an \"actions\" list)")
    return list(acts), ""


def expand_actions(workdir: Path, items: Any, *, field: str = "actions") -> "tuple[Optional[list], str]":
    if isinstance(items, str):
        items = [items]
    if not isinstance(items, list) or not items:
        return None, f"ERROR: '{field}' must be a non-empty list of action strings"
    out: list = []
    for item in items:
        s = str(item).strip()
        if not s.startswith("@"):
            out.append(item)
            continue
        acts, err = load_plan_file(workdir, s[1:])
        if err:
            return None, err
        if any(str(a).strip().startswith("@") for a in acts):
            return None, f"ERROR: {s[1:].strip()!r} refers to another file; a file must hold plain actions"
        out.extend(acts)
    return out, ""


def render_board(obs: Optional[str], *, rulers: bool = False,
                 region: Optional[list] = None) -> str:
    text = obs or ""
    lines = text.splitlines()
    if region:
        try:
            x0, y0, x1, y1 = (int(v) for v in region)
        except (TypeError, ValueError):
            return "ERROR: region must be [x0, y0, x1, y1] (character coordinates)"
        lines = [ln[x0:x1 + 1] for ln in lines[y0:y1 + 1]]
        ox, oy = x0, y0
    else:
        ox = oy = 0
    if not rulers:
        return "\n".join(lines)
    width = max((len(ln) for ln in lines), default=0)
    tens = "    " + "".join(str(((ox + i) // 10) % 10) if (ox + i) % 10 == 0 else " "
                            for i in range(width))
    ones = "    " + "".join(str((ox + i) % 10) for i in range(width))
    body = [f"{oy + i:3d} {ln}" for i, ln in enumerate(lines)]
    return "\n".join([tens, ones, *body])


def render_observation(agent: Any, latest, legal: list) -> str:
    st = latest.state
    parts = [
        f"Room {st.room} | camera view={st.view} yaw={st.yaw} | "
        f"gems {st.gems}/90 | rooms visited {len(st.visited_rooms)} | actions used {st.action_count}",
    ]
    if st.transition:
        parts.append(st.transition)
    if st.dead:
        parts.append(f"DEAD: {st.death_message}")
    if getattr(agent, "last_outcome", ""):
        parts.append(f"Last turn: {agent.last_outcome}")
    parts += [
        "",
        "Legal actions: " + ", ".join(legal),
        "Rooms you have visited (go to level accepts only these): "
        + ", ".join(sorted(st.visited_rooms)),
        "",
        "Board:",
        render_board(st.observation),
        "",
        f"World model: {'installed' if getattr(agent, 'world', None) is not None else 'NONE yet'}"
        f"; recorded transitions: {len(getattr(agent, 'timeline', []))}",
        "",
    ]
    parts.append("Decide the next action(s), then end by calling commit_actions.")
    return "\n".join(parts)


_WORLD_FACTS = """\
THE GAME (this is everything the benchmark tells any player):
- You are exploring an unknown 3D world made of rooms, rendered to you as ASCII text. Each tile of
  the world is drawn as a group of 4x4 characters, and a room's picture is at most 64x64 characters.
- The character `P` is the player (you). The character `G` is a gem. EVERY OTHER object is drawn
  with a character that was assigned randomly for this run: the same character means the same kind
  of object within this run, but the character itself tells you nothing about what the object does.
- The picture is an isometric projection of the 3D world. There are 20 camera angles: 5 vertical
  angles (top, top-diagonal, diagonal, side-diagonal, side) x 4 rotations (yaw 0-3).
- MOVEMENT IS SCREEN-RELATIVE: `up` moves the player up on the CURRENT PICTURE.
- The world is a grid of rooms addressed by two letters (for example `level_HxI`). Walking off the
  edge of one room takes you into a neighbouring room.
- YOUR GOAL: enter as many rooms as possible and collect as many gems as possible. There are 90
  gems in total.
- The player can die. When that happens the observation says so, and until the player is alive
  again the ONLY actions accepted are undo, reset and go to level.
$BUDGET

THE ACTIONS (exactly these, nothing else):
- `up`, `down`, `left`, `right` — move one step in that direction ON THE CURRENT PICTURE.
- `rotate camera up`, `rotate camera down` — change the camera's vertical angle.
- `rotate camera left`, `rotate camera right` — rotate the camera (yaw).
- `undo` — undo your most recent movement. Gems you already collected stay collected.
- `reset` — reset the current room to its entry state. Gems you already collected stay collected.
- `go to level X Y` — return to a room you have ALREADY VISITED (X and Y are its two letters).
- Whether a movement was blocked is NEVER reported. Infer the effect of an action only from the
  resulting board.
"""

_CODE_CONTRACT = """\
YOUR WORLD MODEL — the one hard requirement of this run:
Write an EXECUTABLE model of this game as Python, in `world_model.py`, defining ONE of:

    def step(obs, action, meta):              # stateless
        return next_obs, info                 # (or just next_obs)

    def predict(state, obs, action, meta):    # stateful — use this if anything invisible matters
        return next_obs, info, next_state     # plus optional init_state(entry_obs, meta, prev_state)

- `obs` / `next_obs` are the BOARD STRING exactly as you receive it. A prediction counts as correct
  only if it matches the real next board CHARACTER FOR CHARACTER.
- `action` is one of the action strings above, verbatim (`"up"`, `"rotate camera left"`, `"undo"`,
  `"reset"`, `"go to level H I"`). `undo`, `reset` and `go to level X Y` are actions like any
  other: your model predicts their results too.
- `meta` is a dict with what you can see on screen: `room`, `view`, `yaw`, `gems`,
  `visited_rooms`.
- `info` is a dict of outcome flags, all optional and False by default:
      gem           this action collected a gem
      room_changed  after this action the player is in a different room
      dead          the player died
      won           the game is won
- A room change, `go to level`, `reset` or death starts a new segment, and the harness calls
  `init_state(entry_obs, meta, prev_state)`: `prev_state` is your own state right after the action
  that ended the previous segment (None at the very start), so carry over anything invisible that
  persists instead of rebuilding it.
- Preloaded globals: `np` (numpy), `ENTRY_OBS` (the board this segment started from), `ENTRY_META`,
  `WORKDIR`.
- You may write as many .py files as you like in the workdir and import them from each other. Only
  `world_model.py` is the world model: the harness always calls ITS `step` / `predict`.
- Imports are limited to your own files plus a standard-library subset (numpy, math, collections,
  itertools, functools, heapq, re, string, copy, dataclasses, typing, enum, operator, bisect, json,
  textwrap). There is no file, network or process access from inside the model.
- `current_board.py` in the workdir always holds the board you are looking at now (`BOARD`) and its
  `META`. It is read-only: `cp` it to another name to keep that board for your code to import later.
  It also holds `MODEL_VERSION`, which identifies the world-model code now installed
  (`world_model.py` plus the files it has imported). A search result or search checkpoint you
  saved is only valid for the model version that produced it. `current_state.pkl` holds your
  model's state for that board (a pickle, written when the state can be pickled).

- OPTIONAL planning helpers in `world_model.py`, used only by `run_bfs` (never scored):
      def is_bfs_goal(state, obs, info, args) -> bool   # your goal; `args` is run_bfs's goal_args
      def bfs_heuristic(state, obs, args) -> float      # your estimate of actions remaining
      def bfs_moves(state, obs, args) -> list           # action sequences to try from this state
      def bfs_state_key(state, obs, args)               # what makes two states the same, any value
  `run_bfs` calls `is_bfs_goal` on every transition it predicts (`state` is None for a stateless
  model). If you define `bfs_heuristic`, `run_bfs` searches with A* instead of breadth-first search.
  A heuristic does not always make a search faster: when the way to the goal first leads away from
  what the heuristic rewards, A* can take longer than breadth-first search.
  If you define `bfs_moves`, each state is expanded with the sequences it returns (a list whose
  items are action strings or lists of them) instead of single actions; every step is still
  predicted, and the plan comes back as single actions. If you define `bfs_state_key`, states that
  return the same value (in the same room and camera angle) are searched only once; any value works,
  lists and dicts included. A search that uses them can only find what they let it reach.
  A search holds a separate copy of your model's state for every state it has yet to expand, so its
  speed and how many states it can hold depend on the size of that state: keep in the state only
  what actions can change.
"""

_METHOD_STEP2 = (
    "2. Build the model in `world_model.py` and CHECK IT AGAINST HISTORY with `backtest`: it replays\n"
    "   your model over every transition recorded so far and reports, per action type, where the\n"
    "   predicted board differs from what actually happened. A model that cannot reproduce the past will\n"
    "   not predict the future.\n")
_METHOD_STEP3 = (
    "3. When the backtest is green, the model is more reliable: `run_bfs` searches inside it for the\n"
    "   shortest action sequence to a goal you define (`is_bfs_goal`). Commit the plan it finds.\n")
_DURABLE_LINE_ROOMS = (
    "- Keep DURABLE memory in files: `world_model.py`, your helper modules, `notes.md` for your\n"
    "  overall notes, and `rooms/<room>.md` for each room (the harness creates it the first time you\n"
    "  enter that room). Notes are not shown to you automatically: read them with `read_file` when you\n"
    "  need them. This conversation is auto-compacted as it grows; files and `read_history` are what\n"
    "  survive.\n")

_UNVERIFIED_MARK = "   as settled in your model; until then, mark it as unverified with a comment in your code.\n"

_METHOD = (
    "Your method (learn the rules, then exploit them):\n"
    "1. LOOK before you move. `inspect_board` prints the board with x/y rulers, crops a region, or diffs\n"
    "   two recorded boards character by character.\n"
    + _METHOD_STEP2 + _METHOD_STEP3 +
    "4. Guess boldly, verify carefully. On one hand, act on hunches: when something looks slightly\n"
    "   different from before, a detail seems a little suspicious, or a new state might be worth reaching,\n"
    "   go there and try it, even without a specific purpose. You may find what you overlooked and solve\n"
    "   problems you had not thought of. On the other hand, make the uncertain parts of your model\n"
    "   certain: anything you inferred rather than directly observed or tried is uncertain, even when\n"
    "   your model reproduces every recorded transition. Take actions that test it before you treat it\n"
    + _UNVERIFIED_MARK +
    "5. Model GENERAL mechanisms. Your code must express rules that hold everywhere in the world. Never\n"
    "   special-case particular rooms, positions, boards or action sequences, and do not keep adding\n"
    "   patches that only fix the transitions in front of you. Room layouts you record are data; the rules\n"
    "   that act on them must be the same everywhere. When something does not fit, work out the general\n"
    "   mechanism behind it and change the model so that this case and every earlier one follow from the\n"
    "   same rules. If how you read the board or how you represent the world state cannot express that\n"
    "   mechanism, do not hesitate to restructure the model substantially.\n"
    "\n"
    + _DURABLE_LINE_ROOMS)


def _budget_line(max_actions: "int | None") -> str:
    if max_actions and max_actions < 100_000:
        return f"- You may use at most {int(max_actions)} game actions in this run."
    return "- There is no step limit and no time limit."


def build_system_prompt(max_actions: "int | None") -> str:
    facts = _WORLD_FACTS.replace("$BUDGET", _budget_line(max_actions))
    return "\n".join([facts, _CODE_CONTRACT, _METHOD]).strip()


_ACTION_HELP = ("one of: up, down, left, right, rotate camera up, rotate camera down, "
                "rotate camera left, rotate camera right, undo, reset, go to level X Y")

_COMMIT_SPEC = {
    "name": "commit_actions",
    "description": (
        "TERMINAL: end deliberation and execute these actions on the game, in order — they form a "
        "queue. Commit as many as your world model lets you predict with confidence; when unsure, "
        "commit one or a series of exploratory actions and look at what happened; states and positions "
        "where you are unsure how your code models the environment and its transition mechanics are the "
        "most worth trying. Each executed action is recorded "
        "as a transition you can replay later."),
    "input_schema": {
        "type": "object",
        "properties": {
            "actions": {"type": "array", "items": {"type": "string"},
                        "description": f"action strings, {_ACTION_HELP}." + AT_FILE_HELP},
            "reason": {"type": "string", "description": "why this plan (one or two sentences)"},
            "suggestion": {"type": "string",
                           "description": "optional note to your future self for the next turn"},
        },
        "required": ["actions", "reason"],
    },
}

_BACKTEST_SPEC = {
    "name": "backtest",
    "description": (
        "Replay world_model.py over the transitions recorded so far and report where it is "
        "wrong. For every recorded action it compares (1) the predicted next board against the "
        "real one, character for character, and (2) the info flags gem / room_changed / dead / won. "
        "Results are broken down by action type (move, camera, undo, reset, goto) so you can see "
        "which part of the model is wrong. This is the only way to know whether your model is "
        "right; a model that fails here cannot be trusted to plan with."),
    "input_schema": {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "description": "only the last N transitions (default: all)"},
            "room": {"type": "string", "description": "only transitions recorded in this room"},
            "kind": {"type": "string",
                     "description": "only this action type: move / camera / undo / reset / goto"},
            "show": {"type": "integer",
                     "description": "how many failing cases to print in full (default 3)"},
        },
    },
}

_BFS_DESC = (
    "Search INSIDE your world model for the shortest action sequence to a goal you define: "
    "is_bfs_goal(state, obs, info, args) in world_model.py, called on every predicted "
    "transition with args = the goal_args you pass here. Branches your model marks dead are "
    "pruned; a branch stops at a transition your model marks room_changed (which can still be the "
    "goal) unless you pass cross_rooms. It runs plain BFS, or A* if your world model defines "
    "bfs_heuristic (a failed A* search lists the closest states it reached, each with a plan file and "
    "the moves there your model predicts change nothing but that were never actually tried); it also "
    "uses your bfs_moves and bfs_state_key if you define them. A found plan (with `after` in front) is ready "
    "for commit_actions and is saved to plans/last_bfs.json (removed once you execute actions). On failure it says why: 'exhausted' "
    "(no state the search could reach satisfies the goal), 'depth_cap' (raise max_depth), "
    "'node_cap' / 'timeout' (the space is too large: sharpen bfs_heuristic, narrow `actions`, pick a "
    "nearer goal, or call again with resume=true), 'memory_cap' (the pending states filled the memory "
    "set aside for searching); running out of budget says nothing about reachability. The call "
    "returns as soon as the search ends.")

_BFS_SPEC = {
    "name": "run_bfs",
    "description": _BFS_DESC,
    "input_schema": {
        "type": "object",
        "properties": {
            "goal_args": {"type": "object",
                          "description": "passed to your is_bfs_goal / bfs_heuristic as `args`"},
            "actions": {"type": "array", "items": {"type": "string"},
                        "description": "action set to search over when you define no bfs_moves "
                                       "(default: the four moves)"},
            "max_depth": {"type": "integer", "description": f"max plan length (default {_BFS_DEFAULT_DEPTH})"},
            "max_nodes": {"type": "integer", "description": f"max nodes to expand (default {_BFS_DEFAULT_NODES})"},
            "time_budget_s": {"type": "number",
                              "description": f"seconds to search (default {_BFS_DEFAULT_TIME_S:.0f}, at "
                                             f"most {_LONG_TOOL_MAX_S:.0f})"},
            "after": {"type": "array", "items": {"type": "string"},
                      "description": "search from the state your model predicts after these "
                                     "actions." + AT_FILE_HELP},
            "resume": {"type": "boolean",
                       "description": "continue your last run_bfs call where it ran out of budget "
                                      "(same arguments). Only while your world model is unchanged; "
                                      "after a model change it starts over and says so."},
            "from_board": {"type": "string",
                           "description": "a .py file in your workdir holding BOARD and META (e.g. a "
                                          "copy of current_board.py): search from that board instead "
                                          "of from where you are. The plan is hypothetical until you "
                                          "actually stand in that state."},
            "cross_rooms": {"type": "boolean",
                            "description": "keep searching past transitions your model marks "
                                           "room_changed, for routes through several rooms (raise "
                                           "max_depth for long routes)."},
        },
    },
}

_MODEL_PREDICT_SPEC = {
    "name": "model_predict",
    "description": (
        "Run your world model forward FROM THE CURRENT STATE and show what it predicts: one "
        "`action`, or a sequence `actions`. A sequence stops at the first action your model "
        "predicts leaves the board unchanged, kills the player, changes the room or wins, and says "
        "which. Use it to check an idea or a plan before committing it. It always starts from where "
        "you are now — it cannot be pointed at a past state."),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "description": _ACTION_HELP},
            "actions": {"type": "array", "items": {"type": "string"},
                        "description": "a sequence, predicted one action after another."
                                       + AT_FILE_HELP},
            "show_board": {"type": "boolean",
                           "description": "print the predicted board (for a sequence: the last one) "
                                          "in full (default true)"},
        },
    },
}

_INSPECT_SPEC = {
    "name": "inspect_board",
    "description": (
        "Look at a board closely. mode='board' prints it with x/y rulers (optionally cropped to a "
        "region), mode='tiles' shrinks it to one character per 4x4 block (the block's top-left "
        "character) for an overview, mode='counts' tallies how many times each character occurs, "
        "mode='diff' shows "
        "every character that changed between the two sides of a recorded transition. Source "
        "'current' is the board you are looking at now; source 'history' takes index= (a "
        "transition number from read_history) and which='before'|'after'."),
    "input_schema": {
        "type": "object",
        "properties": {
            "source": {"type": "string", "description": "current (default) / history"},
            "index": {"type": "integer", "description": "history: which transition"},
            "which": {"type": "string", "description": "history: before / after (default after)"},
            "mode": {"type": "string", "description": "board (default) / tiles / counts / diff"},
            "region": {"type": "array", "items": {"type": "integer"},
                       "description": "[x0, y0, x1, y1] character coordinates"},
        },
    },
}

_HISTORY_SPEC = {
    "name": "read_history",
    "description": (
        "The recorded ground truth: every action you executed, with the board before and after it "
        "and its outcome flags. This is what the backtest replays, and it survives context "
        "compaction. Use last=N, index=i, room=..., kind=... or flag filters to narrow it down; "
        "boards=false gives a compact list without the full pictures." + "$HISTORY_FILE_NOTE"),
    "input_schema": {
        "type": "object",
        "properties": {
            "last": {"type": "integer", "description": "the last N transitions"},
            "index": {"type": "integer", "description": "one transition by number"},
            "start": {"type": "integer"}, "end": {"type": "integer"},
            "room": {"type": "string", "description": "only this room"},
            "kind": {"type": "string", "description": "move / camera / undo / reset / goto"},
            "flag": {"type": "string", "description": "only steps with this flag: " + " / ".join(FLAG_NAMES)},
            "boards": {"type": "boolean", "description": "include the full boards (default true)"},
        },
    },
}

_FILE_SPECS = [
    {"name": "read_file",
     "description": "Read a file from your workdir.",
     "input_schema": {"type": "object",
                      "properties": {"path": {"type": "string"},
                                     "start": {"type": "integer"}, "end": {"type": "integer"}},
                      "required": ["path"]}},
    {"name": "write_file",
     "description": ("Write a file in your workdir, replacing it. Writing world_model.py "
                     "compiles and installs it as your live world model; writing any other .py "
                     "file makes it importable from the model."),
     "input_schema": {"type": "object",
                      "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                      "required": ["path", "content"]}},
    {"name": "edit_file",
     "description": ("Replace an exact substring in a file (old_string must occur exactly once — "
                     "include surrounding context to disambiguate; pass replace_all=true to replace "
                     "every occurrence instead). Prefer this over rewriting a whole file."),
     "input_schema": {"type": "object",
                      "properties": {"path": {"type": "string"}, "old_string": {"type": "string"},
                                     "new_string": {"type": "string"},
                                     "replace_all": {"type": "boolean",
                                                     "description": "replace every occurrence (default false)"}},
                      "required": ["path", "old_string", "new_string"]}},
    {"name": "grep",
     "description": "Search your workdir files for a regular expression.",
     "input_schema": {"type": "object",
                      "properties": {"pattern": {"type": "string"}, "path": {"type": "string"},
                                     "glob": {"type": "string"}},
                      "required": ["pattern"]}},
    {"name": "find",
     "description": "List files in your workdir.",
     "input_schema": {"type": "object",
                      "properties": {"path": {"type": "string"}, "glob": {"type": "string"}}}},
]

_FILE_MGMT_SPECS = [
    {"name": "cp",
     "description": ("Copy a file inside your workdir. "
                     "Parent directories are created; an existing destination is overwritten. "
                     "Copying onto world_model.py installs it as your live world model."),
     "input_schema": {"type": "object",
                      "properties": {"src": {"type": "string", "description": "source path"},
                                     "dst": {"type": "string", "description": "destination under your workdir"}},
                      "required": ["src", "dst"]}},
    {"name": "mv",
     "description": ("Move or rename a file or directory inside your workdir. Parent directories "
                     "are created. Moving onto world_model.py installs it as your live world "
                     "model. Run records and harness directories are refused."),
     "input_schema": {"type": "object",
                      "properties": {"src": {"type": "string", "description": "path under your workdir"},
                                     "dst": {"type": "string", "description": "new path under your workdir"}},
                      "required": ["src", "dst"]}},
    {"name": "rm",
     "description": ("Delete a file under your workdir. To delete a directory and everything in it, "
                     "pass recursive=true. Run records and harness directories are refused."),
     "input_schema": {"type": "object",
                      "properties": {"path": {"type": "string", "description": "file or directory under your workdir"},
                                     "recursive": {"type": "boolean",
                                                   "description": "allow deleting a directory and its contents (default false)"}},
                      "required": ["path"]}},
]

_EXEC_TIMEOUT_DESC = (f"seconds (default {_EXEC_DEFAULT_TIMEOUT_S:.0f}, at most {_LONG_TOOL_MAX_S:.0f}); "
                      "the call returns as soon as the process exits")

_EXEC_SPECS = [
    {"name": "run_python",
     "description": ("Run a Python snippet or file in a scratch subprocess inside the workdir. "
                     "It has no access to the game and no network."),
     "input_schema": {"type": "object",
                      "properties": {"code": {"type": "string"}, "path": {"type": "string"},
                                     "timeout": {"type": "number", "description": _EXEC_TIMEOUT_DESC}}}},
    {"name": "run_shell",
     "description": "Run a shell command inside the workdir. No game access, no network.",
     "input_schema": {"type": "object",
                      "properties": {"command": {"type": "string"},
                                     "timeout": {"type": "number", "description": _EXEC_TIMEOUT_DESC}},
                      "required": ["command"]}},
]


def exec_sandbox_binary() -> str:
    if sys.platform != "linux":
        raise RuntimeError("MazeBench execution tools require Linux and bubblewrap (bwrap).")
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        raise RuntimeError(
            "bubblewrap (bwrap) is required for run_python / run_shell; "
            "install it first (Debian/Ubuntu: sudo apt-get install bubblewrap).")
    return bwrap


def sandbox_cpu_count() -> int:
    raw = os.environ.get("SLURM_CPUS_PER_TASK", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


_CPU_NOTE_PY = (" The machine has {n} CPU cores; multiprocessing works with the 'fork' start method. If a "
                "search you wrote yourself cannot find what you need, consider optimizing it, for example "
                "by using several processes.")
_CPU_NOTE_SH = " The machine has {n} CPU cores."

_HISTORY_NOTE = " The same records, one JSON line per action, are in history.jsonl for your scripts."


def tool_specs() -> list:
    hist = copy.deepcopy(_HISTORY_SPEC)
    hist["description"] = hist["description"].replace("$HISTORY_FILE_NOTE", _HISTORY_NOTE)
    specs = [copy.deepcopy(_COMMIT_SPEC), copy.deepcopy(_INSPECT_SPEC), hist,
             copy.deepcopy(_BACKTEST_SPEC), copy.deepcopy(_BFS_SPEC), copy.deepcopy(_MODEL_PREDICT_SPEC)]
    specs += copy.deepcopy(_FILE_SPECS) + copy.deepcopy(_FILE_MGMT_SPECS)
    ncpu = sandbox_cpu_count()
    for s in copy.deepcopy(_EXEC_SPECS):
        s["description"] += (_CPU_NOTE_PY if s["name"] == "run_python" else _CPU_NOTE_SH).format(n=ncpu)
        specs.append(s)
    return sorted(specs, key=lambda s: s["name"])


class ToolBox:

    _MODEL_EXEC_TOOLS = frozenset({"backtest", "run_bfs", "model_predict"})

    def __init__(self, agent: Any, obs: Optional[str], legal: list) -> None:
        self.agent = agent
        self.obs = obs
        self.legal = [str(a) for a in (legal or [])]

    def dispatch(self, name: str, args: dict) -> str:
        fn = getattr(self, f"tool_{name}", None)
        if fn is None:
            return f"ERROR: unknown tool {name!r}"
        budget = _BFS_TOOL_BUDGET_S if name == "run_bfs" else _TOOL_BUDGET_S
        from . import call_clock
        queued = call_clock.queued_s()
        room = _CALL_WINDOW_S - queued
        max_hours = getattr(self.agent, "max_hours", None)
        t0 = getattr(self.agent, "_t0", None)
        if max_hours and t0:
            run_room = float(t0) + float(max_hours) * 3600.0 + _RUN_GRACE_S - time.time()
            if run_room <= 0:
                return ("ERROR: this run is pausing now. End your turn: commit the actions you have, or "
                        "reply with one line.")
            room = min(room, run_room)
        if room < _MIN_CALL_ROOM_S:
            return (f"ERROR: {name} did not run: it waited {queued:.0f}s behind other tool calls and "
                    "no time was left for it. Call it again on its own.")
        if budget > 0:
            budget = min(budget, room)
        self._exec_room = room
        use_alarm = (name in self._MODEL_EXEC_TOOLS and budget > 0
                     and threading.current_thread() is threading.main_thread())
        self._tool_deadline = (time.monotonic() + budget) if use_alarm else None

        class _Budget(BaseException):
            pass

        timed_out = f"ERROR: {name} timed out after {int(budget)}s."
        prev = None
        try:
            if use_alarm:
                def _on_alarm(_sig, _frm):
                    raise _Budget()
                prev = signal.signal(signal.SIGALRM, _on_alarm)
                signal.alarm(int(budget))
            try:
                return fn(args)
            except _Budget:
                return timed_out
            except Exception as e:
                return f"ERROR: {type(e).__name__}: {e}"
            finally:
                if use_alarm:
                    try:
                        signal.alarm(0)
                    except _Budget:
                        pass
        except _Budget:
            return timed_out
        finally:
            if prev is not None:
                signal.signal(signal.SIGALRM, prev)

    def _resolve(self, path: Any, *, write: bool) -> Optional[Path]:
        raw = str(path or "").strip()
        if not raw:
            return None
        p = Path(raw)
        if not p.is_absolute():
            p = self.agent.workdir / p
        p = p.resolve()
        wd = self.agent.workdir.resolve()
        if write:
            if not p.is_relative_to(wd):
                return None
            if p.name in _PROTECTED:
                return None
            if p in (wd / CURRENT_BOARD_FILE, wd / HISTORY_FILE):
                return None
            rel = p.relative_to(wd).parts
            if rel and rel[0] in _READONLY_DIRS:
                return None
            return p
        if p.name in _PROTECTED:
            return None
        if p.is_relative_to(wd):
            return p
        return None

    def _display(self, p: Path) -> str:
        try:
            return str(p.relative_to(self.agent.workdir.resolve()))
        except ValueError:
            return str(p)

    def tool_read_file(self, args: dict) -> str:
        name = Path(str(args.get("path") or "")).name
        if name in _PROTECTED:
            return (f"ERROR: {name} is a run record kept by the harness and is not readable. "
                    "Use read_history for the transitions you have executed.")
        p = self._resolve(args.get("path"), write=False)
        if p is None or not p.is_file():
            return f"ERROR: cannot read {args.get('path')!r}"
        text = p.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        s, e = args.get("start"), args.get("end")
        if s is not None or e is not None:
            s = max(1, int(s or 1)); e = min(len(lines), int(e or len(lines)))
            lines = lines[s - 1:e]
            head = f"{self._display(p)} lines {s}-{e} of {len(text.splitlines())}"
        else:
            head = f"{self._display(p)} ({len(lines)} lines)"
        return head + "\n" + "\n".join(lines)

    def tool_write_file(self, args: dict) -> str:
        p = self._resolve(args.get("path"), write=True)
        content = args.get("content")
        if p is None:
            return self._write_refused(args.get("path"))
        if not isinstance(content, str):
            return "ERROR: 'content' (string) is required"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        note = f"OK: wrote {self._display(p)} ({len(content)} bytes)."
        return note + self._install_if_model(p)

    def tool_edit_file(self, args: dict) -> str:
        p = self._resolve(args.get("path"), write=True)
        old, new = args.get("old_string"), args.get("new_string")
        if p is None:
            return self._write_refused(args.get("path"))
        if not isinstance(old, str) or not isinstance(new, str):
            return "ERROR: old_string and new_string must both be strings"
        if not p.is_file():
            return f"ERROR: {self._display(p)} does not exist"
        text = p.read_text(encoding="utf-8")
        n = text.count(old)
        if n == 0:
            return "ERROR: old_string not found"
        if n > 1 and not self._truthy(args.get("replace_all")):
            return (f"ERROR: old_string occurs {n} times — add surrounding context to disambiguate, "
                    "or pass replace_all=true")
        p.write_text(text.replace(old, new), encoding="utf-8")
        return (f"OK: replaced {n} occurrence{'s' if n != 1 else ''} in {self._display(p)}."
                + self._install_if_model(p))

    _HARNESS_ENTRIES = ("codex_cwd", ".gitignore")

    @staticmethod
    def _truthy(v) -> bool:
        return v is True or str(v).strip().lower() in ("true", "1", "yes")

    @staticmethod
    def _arg(args: dict, *names):
        for n in names:
            v = args.get(n)
            if v not in (None, ""):
                return v
        return None

    def _harness_owned(self, p: Path) -> bool:
        wd = self.agent.workdir.resolve()
        try:
            rel = p.resolve().relative_to(wd).parts
        except ValueError:
            return True
        return bool(rel) and rel[0] in self._HARNESS_ENTRIES

    _MODEL_GONE = (" NOTE: that was your world-model file; the model already loaded stays active for "
                   "now, but write world_model.py again before the run is resumed.")

    def tool_cp(self, args: dict) -> str:
        raw_src = self._arg(args, "src", "source", "from")
        raw_dst = self._arg(args, "dst", "dest", "destination", "to")
        name = Path(str(raw_src or "")).name
        if name in _PROTECTED:
            return (f"ERROR: {name} is a run record kept by the harness and is not readable. "
                    "Use read_history for the transitions you have executed.")
        src = self._resolve(raw_src, write=False)
        if src is None or not src.is_file():
            return f"ERROR: cannot read {raw_src!r} (allowed: a file in your workdir)."
        dst = self._resolve(raw_dst, write=True)
        if dst is not None and dst.is_dir():
            dst = self._resolve(str(dst / src.name), write=True)
        if dst is None or dst == self.agent.workdir.resolve() or self._harness_owned(dst):
            return self._write_refused(raw_dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return f"OK: copied {self._display(src)} → {self._display(dst)}." + self._install_if_model(dst)

    def tool_mv(self, args: dict) -> str:
        raw_src = self._arg(args, "src", "source", "from")
        raw_dst = self._arg(args, "dst", "dest", "destination", "to")
        wd = self.agent.workdir.resolve()
        src = self._resolve(raw_src, write=True)
        if src is None or src == wd or self._harness_owned(src):
            return f"ERROR: refused — you can only move your own files inside your workdir (got {raw_src!r})."
        if not src.exists():
            return f"ERROR: no such file: {raw_src!r}"
        dst = self._resolve(raw_dst, write=True)
        if dst is not None and dst.is_dir():
            dst = self._resolve(str(dst / src.name), write=True)
        if dst is None or dst == wd or self._harness_owned(dst):
            return self._write_refused(raw_dst)
        was_model = src.name == MODEL_FILE and src.parent == wd
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        note = self._install_if_model(dst)
        if was_model and not (dst.name == MODEL_FILE and dst.parent == wd):
            note += self._MODEL_GONE
        return f"OK: moved {self._display(src)} → {self._display(dst)}." + note

    def tool_rm(self, args: dict) -> str:
        raw = self._arg(args, "path", "file_path", "filename", "target")
        wd = self.agent.workdir.resolve()
        p = self._resolve(raw, write=True)
        if p is None or p == wd or self._harness_owned(p):
            return f"ERROR: refused — you can only delete your own files inside your workdir (got {raw!r})."
        if not p.exists():
            return f"ERROR: no such file: {raw!r}"
        if p.is_dir():
            if not self._truthy(args.get("recursive")):
                return f"ERROR: {self._display(p)} is a directory — pass recursive=true to delete it."
            shutil.rmtree(p)
        else:
            p.unlink()
        note = self._MODEL_GONE if (p.name == MODEL_FILE and p.parent == wd) else ""
        return f"OK: deleted {self._display(p)}." + note

    def _write_refused(self, path: Any) -> str:
        return f"ERROR: refused — {path!r} is outside your workdir or is a protected run record."

    def _install_if_model(self, p: Path) -> str:
        if p.name == MODEL_FILE and p.resolve().parent != self.agent.workdir.resolve():
            return (f" (Not installed: only {MODEL_FILE} at the top of your workdir is the live "
                    "world model.)")
        if p.name != MODEL_FILE:
            return " It is importable from your world model." if p.suffix == ".py" else ""
        try:
            desc = self.agent.install_world_model(p.read_text(encoding="utf-8"))
        except Exception as e:
            return f"\nBUT IT DID NOT COMPILE: {type(e).__name__}: {e}"
        return f"\nInstalled as the live world model [{desc}]."

    def tool_grep(self, args: dict) -> str:
        pat = str(args.get("pattern") or "")
        try:
            rx = re.compile(pat)
        except re.error as e:
            return f"ERROR: bad pattern: {e}"
        base = self._resolve(args.get("path") or ".", write=False) or self.agent.workdir
        hits = []
        for f in sorted(base.rglob(str(args.get("glob") or "*"))):
            if not f.is_file() or f.name in _PROTECTED or not self._inside(f):
                continue
            try:
                for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    if rx.search(line):
                        hits.append(f"{self._display(f)}:{i}: {line[:200]}")
            except OSError:
                continue
            if len(hits) > 200:
                break
        return "\n".join(hits[:200]) if hits else "(no matches)"

    def tool_find(self, args: dict) -> str:
        base = self._resolve(args.get("path") or ".", write=False) or self.agent.workdir
        glob = str(args.get("glob") or "*")
        rows = [self._display(f) for f in sorted(base.rglob(glob))
                if f.is_file() and f.name not in _PROTECTED and self._inside(f)]
        return "\n".join(rows[:400]) if rows else "(nothing found)"

    def _inside(self, f: Path) -> bool:
        try:
            return f.resolve().is_relative_to(self.agent.workdir.resolve())
        except OSError:
            return False

    def _board_from(self, args: dict) -> "tuple[Optional[str], str, Optional[str]]":
        source = str(args.get("source") or "current").lower()
        if source == "current":
            return self.obs, "current board", None
        tl = getattr(self.agent, "timeline", [])
        if not tl:
            return None, "", "ERROR: no transitions recorded yet"
        idx = args.get("index")
        i = len(tl) - 1 if idx is None else int(idx)
        if not (0 <= i < len(tl)):
            return None, "", f"ERROR: index {i} out of range (0..{len(tl) - 1})"
        ts = tl[i]
        which = str(args.get("which") or "after").lower()
        obs = ts.before_obs if which == "before" else ts.after
        return obs, f"transition {i} ({ts.action}) {which}", None

    def tool_inspect_board(self, args: dict) -> str:
        mode = str(args.get("mode") or "board").lower()
        if mode == "diff":
            tl = getattr(self.agent, "timeline", [])
            if not tl:
                return "ERROR: no transitions recorded yet"
            i = int(args.get("index", len(tl) - 1))
            if not (0 <= i < len(tl)):
                return f"ERROR: index {i} out of range (0..{len(tl) - 1})"
            ts = tl[i]
            return self._diff(ts.before_obs, ts.after, f"transition {i} ({ts.action})")
        obs, label, err = self._board_from(args)
        if err:
            return err
        if mode == "counts":
            from collections import Counter
            c = Counter(ch for ch in (obs or "") if ch != "\n")
            body = ", ".join(f"{ch!r}={n}" for ch, n in c.most_common())
            return f"{label}: character counts\n{body}"
        if mode == "tiles":
            small = "\n".join(row[::4] for row in (obs or "").splitlines()[::4])
            return (f"{label} — one character per 4x4 block (its top-left character); "
                    "rulers count blocks\n" + render_board(small, rulers=True))
        if mode != "board":
            return f"ERROR: mode must be board / tiles / counts / diff (got {mode!r})"
        return f"{label}\n" + render_board(obs, rulers=True, region=args.get("region"))

    @staticmethod
    def _diff(before: str, after: str, label: str) -> str:
        bl, al = (before or "").splitlines(), (after or "").splitlines()
        if len(bl) != len(al):
            return (f"{label}: the board changed shape, {len(bl)} rows -> {len(al)} rows "
                    "(the camera angle or the room changed).")
        rows = []
        total = 0
        for y, (b, a) in enumerate(zip(bl, al)):
            for x, (cb, ca) in enumerate(zip(b.ljust(len(a)), a.ljust(len(b)))):
                if cb != ca:
                    total += 1
                    if len(rows) < 400:
                        rows.append(f"  ({x},{y}): {cb!r} -> {ca!r}")
        if not total:
            return f"{label}: nothing changed (the two boards are identical)."
        head = f"{label}: {total} character(s) changed (x, y): old -> new"
        if total > len(rows):
            rows.append(f"  … {total - len(rows)} more")
        return head + "\n" + "\n".join(rows)

    def tool_read_history(self, args: dict) -> str:
        tl = list(getattr(self.agent, "timeline", []))
        if not tl:
            return "(no transitions recorded yet)"
        idx = list(range(len(tl)))
        if args.get("index") is not None:
            i = int(args["index"])
            idx = [i] if 0 <= i < len(tl) else []
        else:
            s = int(args.get("start") or 0)
            e = int(args.get("end") if args.get("end") is not None else len(tl) - 1)
            idx = [i for i in idx if s <= i <= e]
            if args.get("last") is not None:
                idx = idx[-int(args["last"]):]
        room = args.get("room")
        kind = args.get("kind")
        flag = args.get("flag")
        def keep(i: int) -> bool:
            ts = tl[i]
            if room and str(ts.before_meta.get("room")) != str(room):
                return False
            if kind and action_kind(ts.action) != str(kind).lower():
                return False
            if flag and not bool(ts.flags.get(str(flag))):
                return False
            return True
        idx = [i for i in idx if keep(i)]
        if not idx:
            return "(no transitions match that filter)"
        boards = args.get("boards", True) is not False and len(idx) <= _BOARD_CAP
        out = [f"{len(idx)} transition(s)" + ("" if boards else " (boards omitted — ask for at most"
                                              f" {_BOARD_CAP} transitions, e.g. last={_BOARD_CAP},"
                                              " to see them in full)")]
        for i in idx:
            ts = tl[i]
            m = ts.before_meta
            flags = [n for n in FLAG_NAMES if ts.flags[n]]
            out.append(f"\n[{i}] action={ts.action!r} room={m.get('room')} view={m.get('view')} "
                       f"yaw={m.get('yaw')} segment={ts.segment}"
                       + (f" flags={','.join(flags)}" if flags else "")
                       + (" REFUSED" if ts.invalid else ""))
            if boards:
                out.append("before:\n" + ts.before_obs)
                out.append("after:\n" + ts.after)
        text = "\n".join(out)
        if len(text) > _HISTORY_BYTES:
            text = (text[:_HISTORY_BYTES]
                    + f"\n… [truncated at {_HISTORY_BYTES} characters — narrow the filter]")
        return text

    def _world(self):
        w = getattr(self.agent, "world", None)
        if w is None:
            raise RuntimeError(f"no world model installed yet — write {MODEL_FILE} first")
        return w

    def tool_backtest(self, args: dict) -> str:
        from .world import backtest_rollout
        world = self._world()
        tl = list(getattr(self.agent, "timeline", []))
        if not tl:
            return "(nothing to backtest yet — no transitions recorded)"
        entries = self.agent.segment_entries()
        idx = [i for i, ts in enumerate(tl) if not getattr(ts, "invalid", False)]
        if args.get("room"):
            idx = [i for i in idx if str(tl[i].before_meta.get("room")) == str(args["room"])]
        if args.get("kind"):
            idx = [i for i in idx if action_kind(tl[i].action) == str(args["kind"]).lower()]
        if args.get("limit"):
            idx = idx[-int(args["limit"]):]
        filtered = bool(args.get("room") or args.get("kind") or args.get("limit"))
        results = backtest_rollout(world, tl, entries,
                                   segments={int(tl[i].segment) for i in idx} if filtered else None,
                                   starts=getattr(self.agent, "seg_starts", None))
        sel = {i: results[i] for i in idx if i in results}
        summary = backtest_summary(sel)
        lines = [f"backtest: {summary['ok']}/{summary['total']} transitions reproduced exactly "
                 f"({summary['accuracy']:.0%})"]
        for kind, s in summary["by_kind"].items():
            lines.append(f"  {kind:7s} {s['ok']:4d}/{s['total']:<4d} ({s['accuracy']:.0%})")
        bad = [i for i in sorted(sel) if sel[i]["errors"]]
        if not bad:
            lines.append("Every recorded transition is reproduced exactly.")
            return "\n".join(lines)
        show = int(args.get("show", 3) or 3)
        lines.append(f"\n{len(bad)} mismatch(es); first {min(show, len(bad))} in full:")
        for i in bad[:show]:
            r = sel[i]
            ts = tl[i]
            lines.append(f"\n--- transition {i}: action={ts.action!r} room={ts.before_meta.get('room')} "
                         f"view={ts.before_meta.get('view')} yaw={ts.before_meta.get('yaw')}")
            for e in r["errors"]:
                lines.append(f"  ! {e}")
            if r["pred"] is not None and r["pred"] != r["after"]:
                lines.append(self._diff(r["pred"], r["after"], "  predicted -> actual"))
        if len(bad) > show:
            lines.append(f"\n… {len(bad) - show} more mismatching transition(s): "
                         + ", ".join(str(i) for i in bad[show:show + 40]))
        return "\n".join(lines)

    def _prefix(self, args: dict) -> "tuple[list, str]":
        from env.maze_env import canonical_action
        after = args.get("after")
        if not after:
            return [], ""
        acts, err = expand_actions(self.agent.workdir, after, field="after")
        if err:
            return [], err
        try:
            return [canonical_action(str(a)) for a in acts], ""
        except ValueError as e:
            return [], f"ERROR: in 'after': {e}"

    def _start_point(self, args: dict):
        from .world import rollout_actions
        prefix, err = self._prefix(args)
        if err:
            return None, None, None, [], err
        state, meta, errs = self.agent.rollout_now()
        if errs:
            return None, None, None, [], ("ERROR: could not roll your model forward to the current "
                                          "state:\n  " + "\n  ".join(errs))
        obs = self.obs
        if prefix:
            obs, state, meta, err = rollout_actions(self._world(), obs, state, meta, prefix)
            if err:
                return None, None, None, [], f"ERROR: {err}"
        return obs, state, meta, prefix, ""

    def _board_start(self, args: dict):
        import ast
        from .world import rollout_actions
        p = self._resolve(args.get("from_board"), write=False)
        if p is None or not p.is_file():
            return None, None, None, [], f"ERROR: from_board: cannot read {args.get('from_board')!r}"
        vals: dict = {}
        try:
            for node in ast.parse(p.read_text(encoding="utf-8")).body:
                if (isinstance(node, ast.Assign) and len(node.targets) == 1
                        and isinstance(node.targets[0], ast.Name)
                        and node.targets[0].id in ("BOARD", "META")):
                    vals[node.targets[0].id] = ast.literal_eval(node.value)
        except (SyntaxError, ValueError) as e:
            return None, None, None, [], f"ERROR: from_board: {e}"
        if not isinstance(vals.get("BOARD"), str) or not isinstance(vals.get("META"), dict):
            return None, None, None, [], ("ERROR: from_board must be a .py file with BOARD = '<board>' "
                                          "and META = {...} (like current_board.py)")
        prefix, err = self._prefix(args)
        if err:
            return None, None, None, [], err
        world = self._world()
        obs, meta = vals["BOARD"], dict(vals["META"])
        state = world.init_state(obs, meta)
        if prefix:
            obs, state, meta, err = rollout_actions(world, obs, state, meta, prefix)
            if err:
                return None, None, None, [], f"ERROR: {err}"
        return obs, state, meta, prefix, ""

    def tool_run_bfs(self, args: dict) -> str:
        from .world import bfs
        want = _BFS_DEFAULT_TIME_S
        if args.get("time_budget_s") is not None:
            try:
                want = float(args.get("time_budget_s"))
            except (TypeError, ValueError):
                want = float("nan")
            if not math.isfinite(want) or want <= 0:
                return "ERROR: time_budget_s must be a positive number of seconds"
        world = self._world()
        from_board = str(args.get("from_board") or "").strip()
        if from_board:
            start_obs, state, meta, prefix, err = self._board_start(args)
        else:
            start_obs, state, meta, prefix, err = self._start_point(args)
        if err:
            return err
        goal_args = args.get("goal_args")
        if isinstance(goal_args, str):
            try:
                goal_args = json.loads(goal_args) if goal_args.strip() else {}
            except ValueError:
                return "ERROR: goal_args must be a JSON object"
        if goal_args is None:
            goal_args = {}
        if not isinstance(goal_args, dict):
            return "ERROR: goal_args must be a JSON object"
        actions = [str(a) for a in (args.get("actions") or ["up", "down", "left", "right"])]
        max_depth = int(args.get("max_depth") or _BFS_DEFAULT_DEPTH)
        max_nodes = int(args.get("max_nodes") or _BFS_DEFAULT_NODES)
        cross = self._truthy(args.get("cross_rooms"))
        version = world.version() if hasattr(world, "version") else ""
        key = json.dumps({"start": hash(start_obs), "room": (meta or {}).get("room"),
                          "view": (meta or {}).get("view"), "yaw": (meta or {}).get("yaw"),
                          "prefix": prefix, "from": from_board, "goal": goal_args,
                          "actions": actions, "cross": cross, "depth": max_depth},
                         sort_keys=True, default=str)
        saved = getattr(self.agent, "_bfs_checkpoint", None)
        resume, head = None, ""
        if self._truthy(args.get("resume")):
            if saved is None:
                head = "(resume: there is no unfinished search to continue — searched from scratch)\n"
            elif saved["version"] != version:
                head = (f"(resume: your world model changed since that search (version "
                        f"{saved['version']} → {version}), so its checkpoint is void — searched "
                        "from scratch)\n")
            elif saved["key"] != key:
                head = ("(resume: the unfinished search had a different start, goal, actions or "
                        "options — searched from scratch)\n")
            else:
                resume = saved["ckpt"]
                head = f"(resumed the unfinished search: {saved['expanded']:,} nodes expanded before)\n"
        harness_clear_dir(Path(self.agent.workdir), self._CLOSEST_DIR)
        room_s = _LONG_TOOL_MAX_S
        deadline = getattr(self, "_tool_deadline", None)
        if deadline is not None:
            room_s = min(room_s, deadline - time.monotonic() - _BFS_TAIL_S)
        budget_s = max(1.0, min(want, room_s))
        if want > budget_s + 0.5:
            head += (f"(time_budget_s={want:.0f} is longer than this run_bfs call may search; it searches "
                     f"for at most {budget_s:.0f}s)\n")
        from .world import memory_heavy, memory_usage, release_memory
        if resume is None:
            self.agent._bfs_checkpoint = None
            saved = None
            release_memory()
        mem0 = memory_usage()
        res = bfs(world, start_obs, meta, actions=actions, goal_args=goal_args, start_state=state,
                  max_depth=max_depth, max_nodes=max_nodes, time_budget_s=budget_s,
                  resume=resume, keep_checkpoint=True, cross_rooms=cross)
        res["memory_before"] = mem0[0] if mem0 else None
        text = head + self._format_bfs(res, actions=actions, max_depth=max_depth, max_nodes=max_nodes)
        if res.get("plan") is None and res.get("near"):
            text += "\n" + self._closest_section(world, res["near"], actions=actions, goal_args=goal_args,
                                                 prefix=prefix, hypothetical=bool(from_board))
        ck = res.get("checkpoint")
        self.agent._bfs_checkpoint = None
        if res.get("termination") == "memory_cap":
            release_memory(close_pool=True)
        heavy = ck is not None and memory_heavy()
        if ck is not None and len(ck["heap"]) <= self._CHECKPOINT_MAX and not heavy:
            now = world.version() if hasattr(world, "version") else ""
            before = saved["expanded"] if (resume is not None and saved) else 0
            self.agent._bfs_checkpoint = {"key": key, "version": now, "ckpt": ck,
                                          "expanded": before + int(res["expanded"])}
            text += (f"\nCheckpoint kept: run_bfs with the same arguments and resume=true continues "
                     f"this search (valid while your world model stays at version {now}).")
        elif ck is not None:
            why = (f"its {len(ck['heap']):,} pending states take too much memory to hold between calls"
                   if heavy else f"more than {self._CHECKPOINT_MAX:,} states were still pending, too many "
                   "to hold")
            ck = res["checkpoint"] = None
            release_memory(close_pool=heavy)
            text += (f"\nNo checkpoint kept: {why}, so resume=true would start over; narrow the search or "
                     "merge states with bfs_state_key.")
        if from_board:
            text = (f"(hypothetical: searched from the board in {from_board}, not from where you are; "
                    "the plan only works once you actually stand in that state)\n" + text)
        if prefix:
            text = (f"(searched from the state your model predicts after your {len(prefix)}-action "
                    "`after` prefix)\n" + text)
        full = (prefix + list(res["plan"])) if res.get("plan") is not None else None
        if prefix and full is not None:
            text += f"\nfull plan including the prefix ({len(full)} actions): " + ", ".join(full)
        if full and from_board:
            return text
        if full:
            if harness_write_json(Path(self.agent.workdir), BFS_PLAN_FILE,
                                  {"actions": full, "goal_args": goal_args}):
                text += (f"\nSaved to {BFS_PLAN_FILE} — commit_actions(actions=[\"@{BFS_PLAN_FILE}\"]) "
                         "commits it without copying the list.")
        return text

    _CHECKPOINT_MAX = 150_000
    _CLOSEST_DIR = "plans/closest"
    _CLOSEST_HISTORY_S = 20.0
    _CLOSEST_MARGIN_S = 10.0

    def _closest_section(self, world, near: list, *, actions: list, goal_args: dict, prefix: list,
                         hypothetical: bool) -> str:
        from .world import _hooks, _unpack, frames_match, tried_actions
        _, _, key_fn = _hooks(world, dict(goal_args or {}), use_h=False, use_moves=False,
                              use_key=bool(getattr(world, "has_state_key", False)))
        rooms = {(e["meta"] or {}).get("room") for e in near}
        singles = [a for a in dict.fromkeys(actions)]
        deadline = getattr(self, "_tool_deadline", None)
        allow = self._CLOSEST_HISTORY_S if deadline is None else min(
            self._CLOSEST_HISTORY_S, deadline - time.monotonic() - self._CLOSEST_MARGIN_S)
        tried, complete = set(), False
        if allow >= 1.0:
            try:
                timeline = list(getattr(self.agent, "timeline", []) or [])
                entries = self.agent.segment_entries() if timeline else {}
                tried, complete = tried_actions(
                    world, timeline, entries, key_fn, rooms, singles, time_budget_s=allow,
                    start_of=self.agent.segment_start if getattr(world, "carries", False) else None)
            except Exception:
                tried, complete = set(), False
        wd = Path(self.agent.workdir)
        lines = [f"Closest states reached (by your bfs_heuristic; {len(near)} shown; plan files are removed once you "
                 "execute actions):"]
        for n, e in enumerate(near, 1):
            m = e["meta"] or {}
            st = _unpack(e["state"])
            quiet = []
            for a in singles:
                try:
                    try:
                        st_a = copy.deepcopy(st)
                    except Exception:
                        st_a = st
                    pred, info, _ = world.predict(st_a, e["obs"], a, dict(m))
                except Exception:
                    continue
                if not info.get("dead") and not info.get("room_changed") and frames_match(pred, e["obs"]) is None:
                    quiet.append(a)
            plan = list(prefix) + list(e["path"])
            where = ""
            if e["depth"] == 0 and hypothetical:
                where = (" — the state after your `after` prefix on that board" if prefix
                         else " — the starting board of this hypothetical search")
            elif not plan:
                where = " — this is where you are now (no actions needed)"
            elif not hypothetical and harness_write_json(wd, f"{self._CLOSEST_DIR}/{n}.json", {"actions": plan}):
                where = f" — plan: @{self._CLOSEST_DIR}/{n}.json"
            lines.append(f"  {n}. bfs_heuristic={e['h']:.1f}, depth {e['depth']}, {m.get('room')}"
                         + where)
            untried = [a for a in quiet if (e["key"], a) not in tried]
            if not quiet:
                lines.append("     no move here is predicted to leave the board unchanged")
            elif untried:
                lines.append("     predicted to change nothing here, never actually tried from this state: "
                             + ", ".join(untried))
            else:
                lines.append("     every move predicted to change nothing here has been tried from this "
                             "state before")
        if not complete:
            lines.append("  (your recorded history could not be fully checked, so some moves listed as "
                         "never tried may have been tried)")
        b = near[0]
        current = not (list(prefix) + list(b["path"])) and not hypothetical
        lines.append("Closest board" + (" (the current board):" if current else ":"))
        lines.append(render_board(b["obs"]))
        return "\n".join(lines)

    _ALGO_DESC = {
        "bfs": "BFS", "astar": "A* with your bfs_heuristic",
        "greedy": "greedy best-first fallback (ordered by your bfs_heuristic)",
        "bfs+novelty": "novelty-pruned BFS fallback",
    }

    @staticmethod
    def _speed_line(r: dict) -> str:
        ms = r.get("predict_ms")
        if not ms:
            return ""
        budget = float(r.get("time_budget_s") or _BFS_DEFAULT_TIME_S)
        return (f"Model speed: predict ≈ {ms:.2f} ms/call → ≈ {int(budget * 1000.0 / ms):,} nodes "
                f"fit in one {budget:.0f}s search.")

    @staticmethod
    def _memory_msg(r: dict) -> str:
        used, limit = r.get("memory") or (0, 0)
        gb = 1024.0 ** 3
        per = ""
        before, pending = r.get("memory_before"), int(r.get("frontier") or 0)
        if before is not None and pending > 0 and used > before:
            per = f", roughly {(used - before) / pending / 1024.0:,.0f} KB each"
        from .world import _mem_fraction
        return (f"memory_cap — the search stops when memory use reaches {_mem_fraction():.0%} of the "
                f"{limit / gb:.0f} GB it may use; it reached {used / gb:.0f} GB with "
                f"{pending} states still pending{per}). The search holds a separate copy of your "
                "model's state for every pending state, so keeping only what actions can change in the "
                "state lets it hold more of them; narrowing `actions` or picking a nearer goal also "
                "helps.")

    def _format_bfs(self, r: dict, *, actions: list, max_depth: int, max_nodes: int) -> str:
        term = r["termination"]
        if term in ("no_goal_capability", "goal_error"):
            return "bfs: cannot search — " + (r["note"] or "; ".join(r.get("errors") or []))
        ph = "; ".join(f"{p['algo']}: {p['expanded']} nodes/{p['secs']:.0f}s → {p['termination']}"
                       for p in (r.get("phases") or []))
        algo = self._ALGO_DESC.get(r.get("algo"), r.get("algo") or "search")
        if r.get("macro"):
            algo += " over your bfs_moves"
        if r.get("custom_key"):
            algo += ", states merged by your bfs_state_key"
        tail: list = []
        if r.get("heuristic_error"):
            tail.append(f"WARNING: bfs_heuristic raised {r['heuristic_error']} — the search "
                        "continued with h=0 from that point.")
        if r.get("errors"):
            tail.append("model errors during search:\n  " + "\n  ".join(r["errors"]))
        sp = self._speed_line(r)
        if sp:
            tail.append(sp)

        has_h = bool(r.get("has_heuristic"))
        if r["plan"] is not None:
            won = " (your model predicts the game is won)" if r.get("goal_reason") == "won" else ""
            lines = [f"bfs: goal reached in {len(r['plan'])} action(s){won} [{algo}]; "
                     f"expanded {r['expanded']} nodes, {r['distinct']} distinct states. "
                     f"Search time: {float(r.get('secs') or 0.0):.1f}s."]
            if not r.get("optimal", True):
                lines.append(
                    f"NOTE: not proven shortest — the main search ran out of budget first ({ph}). "
                    + ("A sharper bfs_heuristic or a nearer goal usually shortens it." if has_h else
                       "Define bfs_heuristic(state, obs, args) to search with A* and get shorter plans."))
            lines.append("(the current board already satisfies your is_bfs_goal — no actions needed.)"
                         if not r["plan"] else
                         f"plan ({len(r['plan'])} actions): " + ", ".join(r["plan"]))
            return "\n".join(lines + tail)

        budget = float(r.get("time_budget_s") or _BFS_DEFAULT_TIME_S)
        advice = (" The space is too large for this search: sharpen bfs_heuristic, narrow `actions`, "
                  "or search to a nearer goal." if has_h else
                  " The space is too large for blind search: define bfs_heuristic(state, obs, args) "
                  "(the same run_bfs call then runs A*), narrow `actions`, or search to a nearer goal.")
        no_proof = " Running out of budget says nothing about whether the goal is reachable."
        spare = budget - float(r.get("secs") or 0.0)
        more = (f" It stopped with {spare:.0f}s of its time budget unused, so raising max_nodes also "
                "buys a deeper search." if term == "node_cap" and spare >= 5.0 else "")
        macro, keyed = bool(r.get("macro")), bool(r.get("custom_key"))
        how = ("through your bfs_moves" if macro else f"with actions {actions}") + (
            ", counting states by your bfs_state_key" if keyed else "") + (
            ", continuing across rooms" if r.get("cross_rooms") else "")
        caveat = (" Your bfs_moves and bfs_state_key decide what this search can reach; a goal they "
                  "leave out would not be found." if macro and keyed else
                  " Your bfs_moves decide what this search can reach; a goal they leave out would "
                  "not be found." if macro else
                  " Your bfs_state_key decides which states count as new; merging states that can "
                  "behave differently would hide the goal." if keyed else "")
        msg = {
            "exhausted": (f"exhausted — all {r['distinct']} states reachable under your model {how} "
                          f"were explored; none satisfies your goal)." + caveat),
            "depth_cap": (f"depth_cap — every path was cut at max_depth={max_depth} before reaching "
                          "the goal). Retry with a larger max_depth."),
            "timeout": f"timeout — the {budget:.0f}s budget ran out)." + advice + no_proof,
            "node_cap": (f"node_cap — hit max_nodes={max_nodes}; {r['frontier']} states still "
                         "pending)." + advice + more + no_proof),
            "memory_cap": self._memory_msg(r) + no_proof,
        }.get(term, f"{term}).")
        lines = [f"bfs: no plan ({msg}",
                 f"[expanded {r['expanded']} nodes, {r['distinct']} distinct states, deepest state "
                 f"reached: depth {r.get('depth_reached', 0)}; {ph}]",
                 f"Search time: {float(r.get('secs') or 0.0):.1f}s."]
        b = r.get("best")
        if b is not None and not r.get("near"):
            lines.append(f"Closest state reached (bfs_heuristic={b['h']:.1f}, depth {b['depth']}): "
                         + (", ".join(b["path"]) if b["path"] else "(the current board)"))
            lines.append(render_board(b["obs"]))
        return "\n".join(lines + tail)

    def tool_model_predict(self, args: dict) -> str:
        from .world import frames_match
        world = self._world()
        action = str(args.get("action") or "").strip()
        seq = args.get("actions")
        if bool(action) == (seq is not None):
            return "ERROR: give exactly one of 'action' or 'actions'"
        if action.startswith("@"):
            return "ERROR: a file of actions goes in 'actions' (e.g. actions=[\"" + action + "\"])"
        if not action:
            return self._predict_sequence(world, seq, args)
        start, state, meta, _, err = self._start_point({})
        if err:
            return err
        pred, info, _ = world.predict(state, start, action, meta)
        out = [f"model_predict({action!r}) from the current board",
               "info: " + json.dumps({k: info.get(k, False) for k in FLAG_NAMES})]
        if args.get("show_board", True) is not False:
            out.append("predicted board:\n" + render_board(pred))
        same = frames_match(pred, start)
        out.append("(the model predicts the board does not change)" if same is None
                   else f"(differs from the current board: {same})")
        return "\n".join(out)

    _SEQ_LIST_CAP = 60

    def _predict_sequence(self, world, seq, args: dict) -> str:
        from env.maze_env import canonical_action
        from .world import advance_meta, frames_match
        acts, err = expand_actions(self.agent.workdir, seq)
        if err:
            return err
        try:
            seq = [canonical_action(str(a)) for a in acts]
        except ValueError as e:
            return f"ERROR: {e}"
        obs, state, meta, _, err = self._start_point({})
        if err:
            return err
        rows, stop = [], ""
        for i, a in enumerate(seq, 1):
            pred, info, state = world.predict(state, obs, a, meta)
            flags = [k for k in FLAG_NAMES if info.get(k)]
            unchanged = frames_match(pred, obs) is None
            rows.append(f"  {i:3d} {a:<20} {'no change' if unchanged else 'changed'}"
                        + (f"  [{', '.join(flags)}]" if flags else ""))
            meta = advance_meta(meta, a, info, state)
            obs = pred
            why = ("kills the player" if info.get("dead") else
                   "changes the room" if info.get("room_changed") else
                   "wins the game" if info.get("won") else
                   "leaves the board unchanged" if unchanged else "")
            if why:
                stop = f"stopped at action {i} ({a!r}): your model predicts it {why}."
                break
        n = len(rows)
        if n > self._SEQ_LIST_CAP:
            rows = rows[:10] + [f"  … {n - 30} more …"] + rows[-20:]
        out = [f"model_predict: {n} of {len(seq)} action(s) predicted from the current board",
               stop or "every action changes the board; no death, room change or win predicted."]
        out += rows
        if args.get("show_board", True) is not False:
            out.append("predicted board after the last predicted action:\n" + render_board(obs))
        return "\n".join(out)

    def _sandbox_wrap(self, cmd: list) -> list:
        bwrap = exec_sandbox_binary()
        import sys as _sys
        wd = str(self.agent.workdir.resolve())
        prefix = str(Path(_sys.executable).resolve().parent.parent)
        argv = [bwrap, "--die-with-parent", "--unshare-all", "--new-session",
                "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
        for ro in ("/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc/alternatives", prefix):
            if Path(ro).exists():
                argv += ["--ro-bind", ro, ro]
        argv += ["--bind", wd, wd]
        for name in _PROTECTED:
            p = Path(wd) / name
            if p.is_file():
                argv += ["--ro-bind", "/dev/null", str(p)]
        for name in (CURRENT_BOARD_FILE, HISTORY_FILE):
            cb = Path(wd) / name
            if cb.is_file() and not cb.is_symlink():
                argv += ["--ro-bind", str(cb), str(cb)]
        argv += ["--chdir", wd, "--setenv", "HOME", wd, "--"]
        return argv + cmd

    def _proc(self, cmd: list, timeout: float) -> str:
        import subprocess
        try:
            cmd = self._sandbox_wrap(cmd)
        except RuntimeError as e:
            return f"ERROR: {e}"
        t0 = time.monotonic()
        try:
            r = subprocess.run(cmd, cwd=str(self.agent.workdir), capture_output=True, text=True,
                               timeout=timeout, env=self.agent.exec_env())
        except subprocess.TimeoutExpired:
            return f"TIMEOUT after {timeout:.0f}s"
        dt = time.monotonic() - t0
        out = (r.stdout or "")[-20000:]
        err = (r.stderr or "")[-4000:]
        return f"exit={r.returncode} in {dt:.1f}s\n{out}" + (f"\nstderr:\n{err}" if err else "")

    @staticmethod
    def _exec_timeout(args: dict) -> float:
        try:
            t = float(args.get("timeout") or _EXEC_DEFAULT_TIMEOUT_S)
        except (TypeError, ValueError):
            t = _EXEC_DEFAULT_TIMEOUT_S
        return min(max(t, 1.0), _LONG_TOOL_MAX_S)

    def tool_run_python(self, args: dict) -> str:
        import sys
        code, path = args.get("code"), args.get("path")
        timeout = min(self._exec_timeout(args), getattr(self, "_exec_room", _LONG_TOOL_MAX_S))
        if path:
            p = self._resolve(path, write=False)
            if p is None or not p.is_file():
                return f"ERROR: cannot run {path!r}"
            return self._proc([sys.executable, str(p)], timeout)
        if not isinstance(code, str) or not code.strip():
            return "ERROR: 'code' or 'path' is required"
        return self._proc([sys.executable, "-c", code], timeout)

    def tool_run_shell(self, args: dict) -> str:
        cmd = str(args.get("command") or "")
        if not cmd.strip():
            return "ERROR: 'command' is required"
        return self._proc(["bash", "-lc", cmd],
                          min(self._exec_timeout(args), getattr(self, "_exec_room", _LONG_TOOL_MAX_S)))
