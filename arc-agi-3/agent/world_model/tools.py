from __future__ import annotations

import functools
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from sandbox.paths import jail_to_host

from . import run_records
from .budgets import BFS_SEARCH_S, TOOL_CAP_S
from .world import CodeWorldModel, bfs, rollout_state

_HEX = "0123456789abcdef"

_REPO_ROOT = Path(__file__).resolve().parents[2]

_LANDLOCK_EXEC = Path(__file__).resolve().parent / "landlock_exec.py"
_SANDBOX_EXEC = shutil.which("sandbox-exec")
_SANDBOX_WARNED: set[str] = set()


@functools.lru_cache(maxsize=1)
def _landlock_ok() -> bool:
    if sys.platform != "linux":
        return False
    try:
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        libc.syscall.restype = ctypes.c_long
        return libc.syscall(444, None, 0, 1) >= 1
    except Exception:
        return False


def _kind_label(stateful) -> str:
    return "stateful (predict)" if stateful else "stateless (step)"


def _contract_label(stateful) -> str:
    return "predict" if stateful else "step"


def _arg(args: dict, *keys: str):
    for k in keys:
        v = args.get(k)
        if v is not None:
            return v
    return None


def _is_noop_sink(path) -> bool:
    return isinstance(path, str) and path.strip() in ("/dev/null",)


def render_grid(grid) -> str:
    if grid is None:
        return "(no grid)"
    g = np.asarray(grid)
    if g.ndim != 2:
        return repr(g.tolist())
    rows = ["".join(_HEX[int(v) & 0xF] for v in row) for row in g]
    return f"shape={g.shape[0]}x{g.shape[1]} (values 0-15 as hex)\n" + "\n".join(rows)


_NOTES_CAP = 4000


def _read_notes(agent: Any, cap: int = _NOTES_CAP) -> str:
    try:
        text = (agent.workdir / "notes.md").read_text(encoding="utf-8").strip()
    except Exception:
        return "(no notes.md yet)"
    if not text:
        return "(notes.md is empty)"
    if len(text) > cap:
        text = text[:cap] + f"\n… (notes.md truncated at {cap} chars — prune it with edit_file)"
    return text


def _world_model_line(agent: Any) -> str:
    return (f"World model: {'installed' if agent.world is not None else 'NONE yet'}; "
            f"history: {len(agent.timeline)} transitions.")


def render_situation(agent: Any, latest, grid, legal: list[int]) -> str:
    needs_xy = "  (action 6 is a click: also give x,y in 0..63)" if 6 in legal else ""
    parts = [
        f"State: {latest.state.name} | level {latest.levels_completed}/{latest.win_levels}",
        f"Legal actions: {legal}{needs_xy}",
        _world_model_line(agent),
        f"Files: workdir (read/write) = {agent.workdir}; "
        f"framework source (read-only) = {agent._src_dir}.",
    ]
    if agent.last_outcome:
        parts.append(f"Last turn: {agent.last_outcome}")
    if agent.last_suggestion:
        parts.append(f"Your note-to-self from last turn (reconsider, don't just obey): {agent.last_suggestion}")
    if agent.last_surprise:
        parts.append(f"NOTE: {agent.last_surprise}")
    parts += [
        "",
        "Your notes (notes.md — maintain it with write_file/edit_file; keep it concise):",
        _read_notes(agent),
        "",
        "Current grid:",
        render_grid(grid),
        "",
        "Decide the next action(s). Update your world model / notes, run a backtest or BFS as "
        "needed, then end by calling commit_actions.",
    ]
    return "\n".join(parts)


def render_observation(agent: Any, latest, grid, legal: list[int]) -> str:
    needs_xy = "  (action 6 is a click: also give x,y in 0..63)" if 6 in legal else ""
    parts = []
    if agent.last_outcome:
        parts.append(f"Result of your last commit: {agent.last_outcome}")
    if agent.last_surprise:
        parts.append(f"NOTE: {agent.last_surprise}")
    parts += [
        f"State: {latest.state.name} | level {latest.levels_completed}/{latest.win_levels}",
        f"Legal actions: {legal}{needs_xy}",
        _world_model_line(agent),
        "",
        "Current grid:",
        render_grid(grid),
        "",
        "Decide the next action(s) (update model/notes, backtest or BFS as needed), then end by "
        "calling commit_actions. If your memory of a rule/layout is fuzzy after a long session, "
        "re-read notes.md / world_model.py / read_history before deciding.",
    ]
    return "\n".join(parts)


def observation_content(agent: Any, latest, grid, legal: list[int], *, continuing: bool) -> list[dict]:
    render = render_observation if continuing else render_situation
    return [{"type": "text", "text": render(agent, latest, grid, legal)}]


SYSTEM_PROMPT = """\
You are an agent playing an unknown 2D grid puzzle game. The grid is a rendered picture; cell values 0-15 are colors. You win a level by reaching its (unknown) goal state; finishing all levels wins the game.

GAME & ACTION PROTOCOL:
- The grid is up to 64x64; values 0-15 are colors/states — a rendered 2D screen, not abstract data.
- Coordinates are (x, y) with (0,0) at the top-left; index a grid as grid[y][x].
- Action ids: RESET is 0; 1,2,3,4,5,7 are simple; 6 is a click carrying x,y in 0..63.
- ACTION0 is a known RESET action (back to the beginning of the level); ACTION1-4 correspond to pressing the visible up, down, left, and right arrow buttons on the screen.
    ACTION5 is a button named SPACE; ACTION6 represents clicking a specified coordinate on the screen; ACTION7 is a button named UNDO.
    These behaviors are not guaranteed — confirm them via transitions.
- Each frame exposes its legal actions. If ACTION6 is legal, no candidate coordinates are
  provided — choosing click points is up to you. In game-over only RESET is valid.
- An action may have intermediate animation frames. Humans see them during gameplay.

Your method (learn the rules, then exploit them):
1. Build a WORLD MODEL as Python code defining ONE of these two functions:

     def step(grid, action, x=None, y=None):
         # grid is a 2D list of ints (0-15). Return the predicted next grid (2D list),
         # or (next_grid, info) where info = {"level_up": bool, "dead": bool, "win": bool}.

     def predict(state, grid, action, x=None, y=None):
         # Same, plus a custom `state` you define (any picklable value; whatever you returned
         # last step is what you receive this step). MUST return (next_grid, info, next_state).
         # You may also define init_state(entry_grid) to provide the initial value — it is
         # called at level start AND on RESET (default {} if omitted).

   Define `predict` OR `step`, not both. The framework rolls your state forward along the REAL
   history (re-initializing at each level entry and each RESET), so you never store anything to
   disk — you only describe the transition. backtest and run_bfs work identically for both
   contracts (they thread state for you).


   GOAL PREDICATES — both optional; either form works: f(grid) or f(state, grid), return bool.
     def is_win_condition(...): your model of the LEVEL-COMPLETION RULE — True exactly on states
       whose entry completes the level. It is PART OF THE WORLD MODEL and backtest CHECKS it
       (mismatch kind win_cond): it must be False on every recorded state that did not complete,
       and True on your own predicted completion grids (coherent with your level_up flags —
       cleanest is to derive the flag from it inside step()/predict()). Encode your win-rule
       hypothesis here instead of leaving it implicit; run_bfs target='advance' stops on it.
     def is_bfs_goal(...): an OPTIONAL self-chosen waypoint ("reach the key") for run_bfs
       target='is_goal' — a planning instrument, never verified, defaults to is_win_condition.
     def bfs_heuristic(...): OPTIONAL, returns a float — your estimate of how many actions remain
       to the goal you search for (0 once it is reached). If defined, run_bfs switches from blind BFS
       to A* automatically (much larger state spaces become searchable) and a failed search reports
       the closest state it reached. A wrong estimate only costs search efficiency, never correctness.
   (A legacy is_goal is still accepted as an unverified goal predicate — prefer the split above.)
   numpy is preloaded as `np`. A global `ENTRY_GRID` is also preloaded: the 2D grid as the
   current level FIRST appeared (the framework sets it per level, and swaps it per level
   during backtest). Ground this level's layout from ENTRY_GRID instead of hard-coding
   coordinates/colours. A global `CURRENT_LEVEL` is also preloaded: the current level number
   (0-based, = levels cleared so far; None if unknown), set/swapped per level alongside
   ENTRY_GRID — read it for per-level branching, or record it in your state inside
   init_state. You may only import: numpy, math, collections, itertools,
   functools, heapq. No file/network/os access (so history reaches your model only via the
   predict() arguments — never try to read files).
   Return level_up/dead/win in info whenever an action completes a level, kills, or wins —
   backtest checks those flags on every step (the next grid of a level-up step is an
   auto-switched new board, so only its flags are scored, not its grid).
2. Rewrite the code ONLY when reality surprised a prediction. Check consistency with
   backtest: a good model predicts every transition's next grid AND its level_up/dead/win
   outcome exactly.
3. When backtest shows the model matches all history, TRUST it: use run_bfs to find a
   path to the goal, then commit that path. run_bfs may itself plan a RESET-first recovery when
   the current state is a dead-end or costly detour. But backtest-green is NOT proven-correct — it only
   matches transitions you have actually walked.
4. When you are unsure, commit an exploratory action that will teach you the most about
   the rules.

Memory: this is ONE CONTINUOUS conversation that spans the whole game — you keep your own
reasoning and what you learned across turns, and each turn you are simply handed the NEXT
observation (what your last committed plan did, the new grid, and legal actions). You do NOT
restart from scratch each turn. BUT as the conversation grows it is AUTO-COMPACTED: the system
first drops older tool outputs, then summarizes older turns — so exact details from far back
(full grids, long tool dumps, precise numbers) may be lost or condensed. Therefore keep your
DURABLE memory in FILES, which survive compaction verbatim: your world-model code + notes.md,
plus the recorded ground-truth history (read_history). After a long stretch, or whenever your
memory of a rule/layout feels fuzzy, RE-READ notes.md / world_model.py / read_history instead
of trusting recollection. Your world model lives in world_model.py in your workdir: read it with read_file,
and writing or editing that file (write_file / edit_file) automatically (re)compiles and
installs it as the live model — there is no separate code tool, and you can now edit it
incrementally (e.g. flip one flag) instead of resubmitting the whole source. If an edit fails
to compile, the file is saved but the previously installed model stays active until you fix it.
Inspect recorded transitions with read_history (ground truth). notes.md is your living
scratchpad — your single most important defense against compaction: record confirmed rules, the
current level's situation, and hypotheses to test, and maintain it with write_file/edit_file as
you learn (it is shown in full at the start of the game and you can re-read it anytime). Keep it
concise and prune stale or disproven entries.

Files: you also have read_file / write_file / edit_file / grep / find / cp / mv / rm. You may
READ your own framework source (the world_model package: agent.py, tools.py, world.py, the
drivers) to learn exactly how you are run and what contracts you must satisfy, and
READ/WRITE/MOVE/DELETE anything under your workdir. Writes, moves and deletes are restricted to
the workdir, and run-critical files (events.jsonl, run.json) are protected. Your notes.md is seeded
in your workdir.

Looking and testing: two tools let you examine the board and your own model.
inspect_grid prints one grid with x/y rulers so you can read exact coordinates (source='current',
a recorded transition, or a level's entry frame), can crop to a region, can count how many cells
hold each value, and can list exactly which cells changed across one recorded transition.
model_predict asks your installed world model what happens NEXT: it predicts one step forward from
the CURRENT frame (a stateful model is handed the state it really has now) and shows the grid it
predicts plus its level_up/dead/win flags. It always steps forward from where the game actually
is — it cannot be pointed at an earlier frame to re-run history. Both tools act on one grid (or
one transition) per call and cannot be scripted, so work through things one step at a time.

Code execution: you also have run_python and run_shell — a REAL Python/shell scratchpad running
in the project's `arc` conda env (numpy and the project deps available; the working dir is your
workdir). This is SEPARATE from your world model:
world_model.py still executes in the restricted sandbox (no os/file/network), whereas
run_python and run_shell are unrestricted real processes. Use them to prototype and UNIT-TEST a
transition rule on concrete numpy arrays before you encode it into step()/predict(), to analyse a
grid or your recorded events.jsonl, or to crunch history — then fold what you confirm back into the
world model and verify with run_backtest. They do NOT share memory with the live game (no current
grid/timeline in-process; reach data via files like events.jsonl) and cannot act on the game — only
commit_actions does that. The GAME'S SOURCE CODE in the repository is OFF-LIMITS to these tools:
do not read, list or import it — infer the game's rules from interaction, not from source.

You MUST end every turn by calling commit_actions with one or more actions to execute. Committing is the only way to act on the game; everything else is deliberation.

STRONGLY RECOMMENDED: before you commit_actions, call backtest to check that your current code reproduces the previously recorded transitions. If it does not yet match the history, fix the model first — committing (especially a multi-step plan) on top of a model that can't even replay the past is unreliable.
"""


TOOL_SPECS: list[dict] = [
    {
        "name": "backtest",
        "description": (
            "Replay the world model (step() or predict()) over all recorded transitions and check "
            "its predictions. For a predict() model it rolls your state forward along "
            "the real history (re-initialized at each level entry and RESET). "
            "Two things are checked: (1) the next GRID, on non-terminal steps only "
            "(a level-up/dead/win step's next frame is an auto-switched new board / end screen, "
            "so its grid is NOT scored); (2) the level_up/dead/win FLAGS from the model's info, on "
            "EVERY step — so the model must correctly predict which action completes a level, "
            "kills, or wins. If is_win_condition is defined it is checked too (kind win_cond): it "
            "must be False on every recorded state that did not complete, and True on your own "
            "predicted completion grids (coherence with your level_up flags). "
            "Resets and the first-ever step are skipped. Output has three parts: "
            "an overall count, the full list of mismatched indices each tagged with the error "
            "kind (grid / level_up / dead / win), and full detail (grids + the model's wrong "
            "prediction) for the most-recent N mismatches. "
            "ENTRY_GRID and CURRENT_LEVEL are swapped to each transition's own level while "
            "replaying, so a general model grounds correctly per level. "
            "By default all transitions are checked — PREFER backtesting against ALL levels: a "
            "regression on a past level means an edit broke a confirmed mechanism. You can still "
            "scope to localise a bug: a contiguous range (start..end), explicit indices, and/or a "
            "single level (level=N or 'current'). Use before trusting run_bfs. "
            "By default the LIVE world model is tested; pass `path` to instead test a CANDIDATE "
            ".py file under your workdir WITHOUT installing it — so you can keep several model "
            "variants as separate files and compare them rather than repeatedly overwriting "
            "world_model.py. The candidate is never installed; a compile error reports back "
            "and leaves the live model untouched."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "OPTIONAL: backtest this candidate .py file (under your workdir) instead of the live model, without installing it. Omit to test the live model.",
                },
                "max_details": {
                    "type": "integer",
                    "description": "how many of the most-recent mismatches to show in full (grids + prediction); default 5, 0 to suppress",
                },
                "start": {
                    "type": "integer",
                    "description": "scope: range start index, inclusive (negatives count from end). Defaults to 0 if only end given.",
                },
                "end": {
                    "type": "integer",
                    "description": "scope: range end index, inclusive (negatives count from end). Defaults to last if only start given.",
                },
                "indices": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "scope: backtest only these transition indices (negatives allowed); takes precedence over range",
                },
                "level": {
                    "description": "scope: backtest only transitions of this level — an int level number, or 'current' for the level being played now (combines with range/indices)",
                },
            },
        },
    },
    {
        "name": "read_history",
        "description": (
            "Inspect/search recorded real transitions (TimeSteps; ground truth: before/action/"
            "after). The output ALWAYS starts with a Summary line of aggregate metadata over the "
            "WHOLE history (level_ups, deaths, wins, resets=action0, clicks=action6, per-action "
            "counts, max_level) — read it to answer 'how many level-ups/resets/...'.\n"
            "Select WHICH transitions: most-recent N (limit), a contiguous range (start..end, "
            "inclusive), or explicit indices (negatives count from end, -1 = last). "
            "Filter by metadata with action / flags / state (combined with AND); filters apply "
            "across the whole history unless a range/indices is also given. "
            "detail='full' (DEFAULT) renders before/after grids; 'brief' gives one summary line "
            "per step (action, coords, #cells changed, state, level, flags) with no grids; "
            "'animation' also renders every intermediate animation frame. "
            "In a contiguous full view a step's before grid is omitted when it equals the "
            "previous step's after (the chain link), so only discontinuities (e.g. after a "
            "reset) show a before grid."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "cap on how many most-recent (matching) steps to show when no range/indices given (default 10)",
                },
                "start": {
                    "type": "integer",
                    "description": "range start index, inclusive (negatives count from end). Defaults to 0 if only end given.",
                },
                "end": {
                    "type": "integer",
                    "description": "range end index, inclusive (negatives count from end). Defaults to the last if only start given.",
                },
                "indices": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "explicit transition indices to show (negatives allowed); takes precedence over range/limit",
                },
                "action": {
                    "description": "filter: keep only steps whose action id matches this int or any int in this list (e.g. 0=reset, 6=click)",
                },
                "flags": {
                    "description": "filter: keep only steps where ALL listed flags are true; allowed 'level_up','dead','win' (string or list)",
                },
                "state": {
                    "description": "filter: keep only steps whose game state matches; e.g. 'WIN','GAME_OVER','NOT_FINISHED' (string or list)",
                },
                "detail": {
                    "type": "string",
                    "enum": ["brief", "full", "animation"],
                    "description": "'full' = before/after grids (DEFAULT); 'brief' = summaries only; 'animation' = before, every intermediate animation frame, and after (prefer one explicit index)",
                },
            },
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read a text file. Allowed: anything under your workdir, and the read-only "
            "world_model framework source. Output is line-numbered; large files are capped "
            "(~50KB) — use offset/limit to page through."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "absolute path, or relative to your workdir"},
                "offset": {"type": "integer", "description": "1-based first line to read (default 1)"},
                "limit": {"type": "integer", "description": "max lines to read (default 2000)"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Create or overwrite a text file with the given content. WRITE IS RESTRICTED to your "
            "workdir (you cannot write to the framework source or anywhere else). Parent dirs are "
            "created as needed. To make an intentional no-op (do nothing), target /dev/null."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "absolute path under your workdir, or relative to it"},
                "content": {"type": "string", "description": "full file contents"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": (
            "Replace an exact substring in a file under your workdir. old_string must occur EXACTLY "
            "once (include surrounding context to disambiguate); pass replace_all=true to replace "
            "every occurrence instead. Workdir-only, like write_file. "
            "(Aliases old_str/new_str are also accepted.) "
            "Target /dev/null for an intentional no-op."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "file under your workdir"},
                "old_string": {"type": "string", "description": "exact text to replace (alias: old_str)"},
                "new_string": {"type": "string", "description": "replacement text (alias: new_str)"},
                "replace_all": {"type": "boolean", "description": "replace every occurrence (default false)"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "cp",
        "description": (
            "Copy a file. Source may be anything readable (your workdir or the read-only "
            "framework source); destination must be under your workdir. Parent dirs are created; "
            "an existing destination is overwritten (except protected run files). If the "
            "destination is world_model.py it is (re)installed as the live world model."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "src": {"type": "string", "description": "source path (workdir or framework source)"},
                "dst": {"type": "string", "description": "destination path under your workdir"},
            },
            "required": ["src", "dst"],
        },
    },
    {
        "name": "mv",
        "description": (
            "Move or rename a file. Both source and destination must be under your workdir "
            "(you cannot move the read-only framework source). Parent dirs are created. Refuses "
            "to move or overwrite run-critical files (events.jsonl, run.json). If the destination "
            "is world_model.py it is (re)installed as the live world model."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "src": {"type": "string", "description": "source path under your workdir"},
                "dst": {"type": "string", "description": "destination path under your workdir"},
            },
            "required": ["src", "dst"],
        },
    },
    {
        "name": "rm",
        "description": (
            "Delete a file under your workdir. Refuses the workdir root and run-critical files "
            "(events.jsonl, run.json). To delete a non-empty directory, pass recursive=true."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "file or directory under your workdir"},
                "recursive": {"type": "boolean", "description": "allow deleting a directory and its contents (default false)"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "grep",
        "description": (
            "Search file contents with a regular expression and return matching lines as "
            "path:line: text. Searches your workdir and the framework source. Capped at 100 matches."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Python regular expression"},
                "path": {"type": "string", "description": "file or directory to search (default: workdir)"},
                "glob": {"type": "string", "description": "only search files matching this glob, e.g. '*.py' (default all)"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "find",
        "description": (
            "List files/dirs matching a glob (recursive), e.g. '*.py' or 'logs/*'. Searches your "
            "workdir and the framework source. Use it to discover what's available before reading."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "glob pattern (default '*')"},
                "path": {"type": "string", "description": "directory to search under (default: workdir)"},
            },
        },
    },
    {
        "name": "run_bfs",
        "description": (
            "Search INSIDE your world model for a shortest action sequence to a goal (reliable "
            "only if backtest passed). Plain BFS by default; if your model defines "
            "bfs_heuristic(grid[, state]) -> estimated actions remaining, the same call runs A* "
            "instead (far larger state spaces become searchable). A goal is a win, a level-up "
            "(advancing one level — search stops there, since the next board auto-switches and the "
            "model can't predict it), a state satisfying your is_win_condition, or "
            "(target='is_goal') your is_bfs_goal waypoint. Death (info.dead) branches are pruned. "
            "On success returns a commit_actions-ready plan plus the predicted final grid (a plan "
            "from the built-in fallback search is labelled as possibly longer than optimal). On "
            "failure it says WHY and what to do next: 'exhausted' (goal unreachable in the model — "
            "suspect the model), 'depth' (raise max_depth), 'budget' / 'timeout' (the space is too "
            "large for blind search: add a bfs_heuristic, narrow actions/clicks, or set a nearer "
            "waypoint; with a heuristic the closest state reached is shown). "
            "It also considers ONE RESET as an optional FIRST step (restart the current level back to "
            "its entry layout) when that yields the shortest path — e.g. the current state is a "
            "dead-end, a costly detour, or out of level budget. Such a plan begins with action 0; it "
            "is never chosen over an equally-short non-reset plan. Disable with allow_reset=false."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "max_depth": {"type": "integer", "description": "max plan length (default 50)"},
                "max_nodes": {"type": "integer", "description": "max nodes to expand (default 20000)"},
                "actions": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "restrict the search to these action ids (default: current legal simple actions; RESET=0 and click=6 are excluded from per-step moves — RESET is offered separately as an optional first step, clicks via the clicks arg)",
                },
                "clicks": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "integer"}},
                    "description": "candidate click targets [[x,y],...] to also branch action 6 on (full 64x64 is infeasible, so you supply the cells worth trying)",
                },
                "target": {
                    "type": "string",
                    "enum": ["advance", "win", "level_up", "is_goal"],
                    "description": "what counts as the goal: 'advance' (default: win OR level_up OR is_win_condition), 'win' (full game win only), 'level_up' (next level only), 'is_goal' (your waypoint: is_bfs_goal, falling back to is_win_condition / legacy is_goal)",
                },
                "allow_reset": {
                    "type": "boolean",
                    "description": "may BFS use ONE RESET (restart current level) as an optional first step when it is the shortest path? (default true)",
                },
            },
        },
    },
    {
        "name": "commit_actions",
        "description": (
            "TERMINAL: end deliberation and execute these actions on the game, in order — they form "
            "a queue. After EACH action the world model is auto-checked against what actually "
            "happened: if it matches, the next queued action runs automatically without consulting "
            "you; the FIRST time reality diverges from the model (or whenever you have no model "
            "yet) the remaining actions are dropped and you are consulted again. So commit a long "
            "plan only when your model is accurate — otherwise just commit one exploratory action. "
            "Each executed step is recorded as a TimeStep. Use action 6 for a click (needs x,y in "
            "0..63). Action 0 is RESET — it restarts the CURRENT level (back to its entry layout, "
            "refunding level budgets); you may commit it (e.g. as the first action) to recover from a "
            "dead-end or wasted detour."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "actions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "integer"},
                            "x": {"type": "integer", "minimum": 0, "maximum": 63},
                            "y": {"type": "integer", "minimum": 0, "maximum": 63},
                        },
                        "required": ["action"],
                    },
                },
                "reason": {
                    "type": "string",
                    "description": "OPTIONAL: why you are committing THESE actions now (justification for this step).",
                },
                "suggestion": {
                    "type": "string",
                    "description": (
                        "OPTIONAL: a handoff to your NEXT turn, which starts from a FRESH context with "
                        "no memory of this turn's reasoning. State your current sub-goal, what you just "
                        "learned, what you RULED OUT and why (so you don't re-litigate it), and the next "
                        "step you intend. Shown back to you next turn as advice to reconsider, not obey."
                    ),
                },
            },
            "required": ["actions"],
        },
    },
    {
        "name": "inspect_grid",
        "description": (
            "Look at one grid up close: printed as hex with x/y rulers so you can read off exact "
            "coordinates, optionally cropped to a region. Pick the grid with `source`: 'current' "
            "(the frame you were just shown), 'history' (a recorded transition — give `index`, and "
            "`which`='before'/'after'), or 'entry' (a level's first frame — give `level`). "
            "mode='grid' (default) prints it; mode='counts' summarises how many cells hold each "
            "value; mode='diff' needs source='history' and lists exactly which cells changed in "
            "that one transition. One grid (or one transition's pair) per call."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "enum": ["current", "history", "entry"],
                           "description": "which grid to look at (default 'current')"},
                "index": {"type": "integer", "description": "transition index for source='history' (negatives count from the end)"},
                "which": {"type": "string", "enum": ["before", "after"],
                          "description": "for source='history': the frame before or after that action (default 'after')"},
                "level": {"type": "integer", "description": "level number for source='entry' (default: the current level)"},
                "region": {"type": "array", "items": {"type": "integer"},
                           "description": "optional crop [x0, y0, x1, y1], inclusive, in grid coordinates"},
                "mode": {"type": "string", "enum": ["grid", "counts", "diff"],
                         "description": "'grid' (default) | 'counts' | 'diff' (history only)"},
            },
        },
    },
    {
        "name": "model_predict",
        "description": (
            "Ask YOUR installed world model what happens NEXT: give an `action` (plus x,y for a "
            "click) and get back the grid it predicts from the CURRENT frame, plus the "
            "level_up/dead/win flags it returned. A stateful model is handed the state it really "
            "has right now (rolled forward along the whole real history). "
            "It always steps forward from where the game actually is — there is no way to point it "
            "at an earlier frame and re-run history. "
            "Pass `path` to run a CANDIDATE .py file under your workdir instead of the live model, "
            "without installing it — so you can try a rule out before committing to it. "
            "show_state=true also prints the hidden state your model returns for that step — when a "
            "stateful model goes wrong it is usually the state, not the rendering."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {"type": "integer", "description": "action id 0-7 (6 is a click, needs x,y)"},
                "x": {"type": "integer", "description": "click x (0..63), only for action 6"},
                "y": {"type": "integer", "description": "click y (0..63), only for action 6"},
                "region": {"type": "array", "items": {"type": "integer"},
                           "description": "optional crop [x0, y0, x1, y1] applied to the printed prediction"},
                "path": {"type": "string",
                         "description": "OPTIONAL: run this candidate .py file (under your workdir) instead of the live model, without installing it"},
                "show_state": {"type": "boolean",
                               "description": "also print your model's hidden state after the step (truncated). Only meaningful for a stateful model"},
            },
            "required": ["action"],
        },
    },
    {
        "name": "run_python",
        "description": (
            "Run real Python in the project's `arc` conda env (which has numpy and the project deps) "
            "and return its combined stdout/stderr + exit code. This is a GENERAL "
            "scratchpad subprocess — full standard library, filesystem and network — and is SEPARATE "
            "from your sandboxed world-model code (world_model.py still runs in the restricted "
            "namespace; this does not). `import numpy as np` etc. work (site-packages); "
            "the working directory is your workdir, so relative "
            "paths and files you wrote land there. The game's source code in the repository is "
            "OFF-LIMITS: do not read, list or import it — infer the rules from interaction, not "
            "from source. Pass EITHER `code` (inline source, may be multi-line) "
            "OR `path` (a .py file under your workdir). Use it to: analyse a grid or events.jsonl with "
            "numpy/scipy, prototype and unit-test a transition rule before encoding it into your world "
            "model, crunch history, or sanity-check a hypothesis. Output is capped (~30KB, head+tail). "
            "NOTE: it does NOT share memory with the live game — current grid/timeline reach it only "
            "via files (e.g. read events.jsonl). It cannot act on the game; only commit_actions can."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "inline Python source to run (multi-line ok). Use this OR path."},
                "path": {"type": "string", "description": "a .py file under your workdir to run instead of inline code"},
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "argv passed to the script (available as sys.argv[1:])",
                },
                "stdin": {"type": "string", "description": "optional text fed to the process's stdin"},
                "timeout": {"type": "number", "description": "seconds before the process is killed (default 60, max 300)"},
            },
        },
    },
    {
        "name": "run_shell",
        "description": (
            "Run a shell command with the `arc` conda env on PATH (so `python`/`pip` resolve to the "
            "arc interpreter) and return its combined stdout/stderr + exit code. The working directory "
            "defaults to your workdir. Use it for quick env/IO chores: "
            "list/inspect files, `python -c ...`, `pip show numpy`, run a script you wrote, etc. "
            "It is a real shell with full filesystem/network access — prefer the "
            "dedicated read_file/write_file/grep/find tools for ordinary file work, and use this for "
            "things they can't do. The game's source code in the repository is OFF-LIMITS: do not "
            "read, list or import it — infer the rules from interaction, not from source. "
            "Output is capped (~30KB, head+tail)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "shell command line (run via `sh -c`)"},
                "cwd": {"type": "string", "description": "working dir under your workdir/framework (default: workdir)"},
                "timeout": {"type": "number", "description": "seconds before the command is killed (default 60, max 300)"},
            },
            "required": ["command"],
        },
    },
]


class ToolBox:

    def __init__(self, agent: Any, grid, legal: list[int]) -> None:
        self.agent = agent
        self.grid = grid
        self.legal = legal

    _MODEL_EXEC_TOOLS = frozenset({"backtest", "run_bfs", "model_predict"})

    def dispatch(self, name: str, args: dict) -> str:
        fn = getattr(self, f"tool_{name}", None)
        if fn is None:
            return f"ERROR: unknown tool {name!r}"
        jailed = self.agent.executor is not None
        budget = TOOL_CAP_S
        use_alarm = (name in self._MODEL_EXEC_TOOLS and not jailed
                     and threading.current_thread() is threading.main_thread())

        class _Budget(Exception):
            pass

        if use_alarm:
            def _on_alarm(signum, frame):
                raise _Budget()
            prev = signal.signal(signal.SIGALRM, _on_alarm)
            signal.alarm(int(budget))
        try:
            return fn(args)
        except _Budget:
            return f"ERROR: {name} timed out after {int(budget)}s."
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"
        finally:
            if use_alarm:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, prev)

    def tool_backtest(self, args: dict) -> str:
        ex = self.agent.executor
        cand_note = ""
        candidate_code = None
        cand_p = None
        raw_path = _arg(args, "path", "file", "file_path")
        if raw_path:
            p = self._resolve(raw_path, write=False)
            if p is None or not p.is_file():
                return f"ERROR: no such .py file under your workdir/framework: {raw_path}"
            candidate_code = p.read_text(encoding="utf-8")
            cand_p = p
            world = None
            if ex is None:
                try:
                    world = CodeWorldModel(candidate_code)
                except Exception as e:
                    return (f"ERROR: candidate {self._display(p)} failed to compile/load "
                            f"({type(e).__name__}: {e}); not backtested, live model unchanged.")
                cand_note = (f"candidate file {self._display(p)} (NOT installed; "
                             f"{_contract_label(world.stateful)}) — ")
        else:
            world = self.agent.world
            if world is None:
                return "ERROR: no world model installed; write world_model.py (write_file) first."
        timeline = self.agent.timeline
        if not timeline:
            return "history is empty; nothing to backtest yet."
        max_details = max(0, int(args.get("max_details", 5) or 0))
        n = len(timeline)

        candidates, explicit, sel_desc = self._select_history_indices(args, n)
        desc_parts: list[str] = [sel_desc] if explicit else []
        if args.get("level") is not None:
            lv = args["level"]
            if isinstance(lv, str) and lv.strip().lower() == "current":
                level = timeline[-1].levels_after
                desc_parts.append(f"level {level} (current)")
            else:
                try:
                    level = int(lv)
                except (TypeError, ValueError):
                    return f"ERROR: level must be an int or 'current', got {lv!r}"
                desc_parts.append(f"level {level}")
            candidates = [i for i in candidates if timeline[i].before_levels == level]
        scope = " ∩ ".join(desc_parts) if desc_parts else "all transitions"

        if ex is not None:
            from sandbox.world_proxy import WorldModelProxy
            proxy = world if isinstance(world, WorldModelProxy) else WorldModelProxy(
                ex, stateful=False, has_is_goal=False)
            bt = proxy.backtest(self.agent.timeline, self.agent.entry_grids, code=candidate_code)
            results = bt["results"]
            if candidate_code is not None:
                cand_note = (f"candidate file {self._display(cand_p)} (NOT installed; "
                             f"{_contract_label(bt['stateful'])}) — ")
        else:
            results = self._backtest_rollout(world)
        total = ok = skipped = 0
        mismatches: list[dict] = []
        for i in candidates:
            r = results.get(i)
            if r is None:
                skipped += 1
                continue
            total += 1
            if r["errors"]:
                mismatches.append({"i": i, "ts": timeline[i], **r})
            else:
                ok += 1

        lines = [
            f"backtest [{cand_note}{scope}]: {ok}/{total} transitions fully correct "
            f"(grid on non-terminal steps + level_up/dead/win flags on EVERY step); "
            f"{len(mismatches)} mismatch(es), {skipped} skipped (resets / no prior grid)."
        ]
        lines.extend(self._backtest_extra_lines(results, candidates, timeline))
        if total == 0:
            lines.append("(no checkable transitions in scope — widen the selection or take a few actions first.)")
            return "\n".join(lines)
        if not mismatches:
            lines.append(
                "Model predicts ALL checkable transitions in scope exactly (grids + level_up/dead/win)"
                " — safe to plan with run_bfs."
            )
            return "\n".join(lines)

        def tag(kinds: list[str]) -> str:
            return "+".join(k for k in ("grid", "level_up", "dead", "win", "win_cond", "error")
                            if k in kinds)

        lines.append(
            "Mismatched transitions (index:error-kind): "
            + ", ".join(f"#{m['i']}:{tag(m['kinds'])}" for m in mismatches)
        )

        if max_details > 0:
            recent = mismatches[-max_details:]
            lines.append(f"\nMost-recent {len(recent)} mismatch(es) in full:")
            for m in recent:
                ts, i = m["ts"], m["i"]
                coord = f"({ts.x},{ts.y})" if ts.x is not None else ""
                aflags = [k for k in ("level_up", "dead", "win") if getattr(ts, k)] or ["none"]
                term_note = "  [terminal: after is auto-switched, grid not scored]" if m["terminal"] else ""
                lines.append(
                    f"\n#{i} action={ts.action_id}{coord}; state={ts.state.name}; "
                    f"level={ts.levels_after}; actual flags={aflags}{term_note}"
                )
                for e in m["errors"]:
                    lines.append(f"  ! {e}")
                lines.append("  before:\n" + render_grid(m["before"]))
                lines.append("  actual after:\n" + render_grid(m["after"]))
                if m["pred"] is None:
                    lines.append("  predicted: (predict()/step() raised — no prediction)")
                else:
                    pflags = [k for k in ("level_up", "dead", "win") if m["info"].get(k)] or ["none"]
                    lines.append(f"  predicted flags={pflags}")
                    if not m["terminal"]:
                        lines.append("  predicted after:\n" + render_grid(m["pred"]))
        return "\n".join(lines)

    @staticmethod
    def _backtest_extra_lines(results: dict, candidates: list, timeline: list) -> list[str]:
        out: list[str] = []
        rows = [(i, results[i]) for i in candidates if results.get(i) is not None]
        ms = [r["predict_ms"] for _, r in rows if r.get("predict_ms")]
        hs = [(i, r["h"]) for i, r in rows if r.get("h") is not None]
        errs = [(i, r["h_error"]) for i, r in rows if r.get("h_error")]
        if hs or errs:
            neg = [(i, h) for i, h in hs if h < 0]
            if errs or neg:
                if errs:
                    what = f"raised {errs[0][1]} at #{errs[0][0]}"
                else:
                    what = f"returned {neg[0][1]:g} (negative) at #{neg[0][0]}"
                out.append(
                    f"bfs_heuristic WARNING: {what} ({len(errs) + len(neg)} such state(s)). A* stays "
                    "correct but falls back to blind ordering there — fix it before relying on run_bfs.")
            else:
                vals = [h for _, h in hs]
                pre = [h for i, h in hs
                       if getattr(timeline[i], "level_up", False) or getattr(timeline[i], "win", False)]
                s = (f"bfs_heuristic: OK — finite and ≥0 on all {len(hs)} states in scope "
                     f"(min {min(vals):g}, max {max(vals):g})")
                if pre:
                    s += (f"; one step before each real level-up it returned "
                          f"{[round(h, 2) for h in pre]} (ideal ≈1 — larger values over-estimate, "
                          "which can lengthen A* plans)")
                out.append(s + ".")
        if ms:
            avg = sum(ms) / len(ms)
            out.append(f"Model speed: predict ≈ {avg:.2f} ms/call → ≈ "
                       f"{int(BFS_SEARCH_S * 1000.0 / avg):,} nodes fit in one "
                       f"{BFS_SEARCH_S / 60:.0f}-min run_bfs.")
        return out

    def _backtest_rollout(self, world: Any = None) -> dict:
        from .world import backtest_rollout
        world = world if world is not None else self.agent.world
        return backtest_rollout(world, self.agent.timeline, self.agent.entry_grids)

    def tool_read_history(self, args: dict) -> str:
        timeline = self.agent.timeline
        if not timeline:
            return "history is empty."
        n = len(timeline)

        candidates, explicit, sel_desc = self._select_history_indices(args, n)
        filt = self._history_filters(args)
        matched = [i for i in candidates if self._match_history(timeline[i], filt)]

        hidden = 0
        if not explicit:
            limit = max(1, int(args.get("limit", 10) or 10))
            if len(matched) > limit:
                hidden = len(matched) - limit
                matched = matched[-limit:]

        detail = str(args.get("detail", "full")).strip().lower()
        show_animation = detail == "animation"
        full = detail != "brief" or bool(args.get("include_grids", False))

        fdesc = self._filter_desc(filt)
        if explicit:
            desc = sel_desc + (f", filtered by {fdesc}" if fdesc else "")
        else:
            desc = (f"most-recent {len(matched)} matching {fdesc}" if fdesc
                    else f"most-recent {len(matched)}")

        plural = "s" if len(matched) != 1 else ""
        hidden_note = f" (+{hidden} older match{'es' if hidden != 1 else ''} hidden; raise limit)" if hidden else ""
        lines = [
            f"{n} transitions total. {self._history_summary(timeline)}",
            f"showing {desc} -> {len(matched)} step{plural}{hidden_note}; "
            f"detail={detail if show_animation else ('full' if full else 'brief')}:",
        ]
        if not matched:
            lines.append("(no transitions match)")
            return "\n".join(lines)

        prev_idx = None
        for i in matched:
            ts = timeline[i]
            ticks = ts.ticks
            b = timeline[i - 1].after if i >= 1 else None
            a = ts.after
            coord = f"({ts.x},{ts.y})" if ts.x is not None else ""
            flags = [k for k in ("level_up", "dead", "win") if getattr(ts, k)]
            if a is None:
                diff_text = "no after grid"
            elif b is None:
                diff_text = "initial (no before)"
            elif b.shape == a.shape:
                diff_text = f"{int((b != a).sum())} cells changed"
            else:
                diff_text = f"shape {tuple(b.shape)}->{tuple(a.shape)}"
            animation_note = (
                f"; +{len(ticks)} intermediate animation frame"
                f"{'s' if len(ticks) != 1 else ''}" if ticks else ""
            )
            lines.append(
                f"#{i} action={ts.action_id}{coord}; {diff_text}; "
                f"state={ts.state.name}; level={ts.levels_after}; "
                f"flags={flags or ['none']}{animation_note}"
            )
            if full:
                if b is None:
                    lines.append("  before: (none — first transition)")
                elif prev_idx == i - 1:
                    lines.append(f"  before: == #{i - 1} after (omitted)")
                else:
                    lines.append("  before:\n" + render_grid(b))
                if show_animation:
                    for tick_index, tick in enumerate(ticks):
                        label = f"#{i} animation {tick_index + 1}/{len(ticks)}"
                        lines.append(f"  {label}:\n" + render_grid(tick))
                lines.append("  after:\n" + render_grid(a))
            prev_idx = i
        return "\n".join(lines)

    @staticmethod
    def _select_history_indices(args: dict, n: int) -> tuple[list[int], bool, str]:

        def norm(v) -> int:
            v = int(v)
            return v + n if v < 0 else v

        raw = args.get("indices")
        if isinstance(raw, list) and raw:
            seen: set[int] = set()
            out: list[int] = []
            for v in raw:
                try:
                    j = norm(v)
                except (TypeError, ValueError):
                    continue
                if 0 <= j < n and j not in seen:
                    seen.add(j)
                    out.append(j)
            out.sort()
            return out, True, f"indices {raw}"

        has_start = args.get("start") is not None
        has_end = args.get("end") is not None
        if has_start or has_end:
            s = norm(args["start"]) if has_start else 0
            e = norm(args["end"]) if has_end else n - 1
            s = max(0, min(s, n - 1))
            e = max(0, min(e, n - 1))
            if s > e:
                s, e = e, s
            return list(range(s, e + 1)), True, f"range #{s}..#{e}"

        return list(range(n)), False, "all"

    @staticmethod
    def _history_summary(timeline: list) -> str:
        acts = Counter(ts.action_id for ts in timeline)
        return (
            "Summary: "
            f"level_ups={sum(1 for ts in timeline if ts.level_up)} "
            f"deaths={sum(1 for ts in timeline if ts.dead)} "
            f"wins={sum(1 for ts in timeline if ts.win)} "
            f"resets(action0)={acts.get(0, 0)} clicks(action6)={acts.get(6, 0)}; "
            f"by-action={{{', '.join(f'{a}:{acts[a]}' for a in sorted(acts))}}}; "
            f"max_level={max((ts.levels_after for ts in timeline), default=0)}"
        )

    @staticmethod
    def _history_filters(args: dict) -> dict | None:
        f: dict = {}
        a = args.get("action")
        if a is not None:
            try:
                f["actions"] = {int(x) for x in (a if isinstance(a, list) else [a])}
            except (TypeError, ValueError):
                pass
        fl = args.get("flags")
        if fl:
            allowed = {"level_up", "dead", "win"}
            got = [x for x in (fl if isinstance(fl, list) else [fl]) if x in allowed]
            if got:
                f["flags"] = got
        st = args.get("state")
        if st:
            f["states"] = {str(x).strip().upper() for x in (st if isinstance(st, list) else [st])}
        return f or None

    @staticmethod
    def _match_history(ts, filt: dict | None) -> bool:
        if not filt:
            return True
        if "actions" in filt and ts.action_id not in filt["actions"]:
            return False
        if "flags" in filt and not all(getattr(ts, fl) for fl in filt["flags"]):
            return False
        if "states" in filt and ts.state.name not in filt["states"]:
            return False
        return True

    @staticmethod
    def _filter_desc(filt: dict | None) -> str:
        if not filt:
            return ""
        parts = []
        if "actions" in filt:
            parts.append(f"action in {sorted(filt['actions'])}")
        if "flags" in filt:
            parts.append(f"flags all of {filt['flags']}")
        if "states" in filt:
            parts.append(f"state in {sorted(filt['states'])}")
        return ", ".join(parts)

    _SKIP_DIR_PARTS = frozenset({"__pycache__", ".git", ".mypy_cache", ".pytest_cache"})

    def _resolve(self, path: Any, *, write: bool) -> Path | None:
        if not isinstance(path, str) or not path.strip():
            return None
        roots = [self.agent.workdir] if write else [self.agent.workdir, self.agent._src_dir]
        try:
            p = Path(path)
            if not write:
                mapped = jail_to_host(p, _REPO_ROOT)
                if mapped is not None:
                    p = mapped
            if not p.is_absolute():
                p = self.agent.workdir / p
            p = p.resolve()
        except Exception:
            return None
        if not any(p == r or p.is_relative_to(r) for r in roots):
            return None
        return p

    def _display(self, p: Path) -> str:
        for root, tag in ((self.agent.workdir, ""), (self.agent._src_dir, "[src] ")):
            try:
                return tag + str(p.relative_to(root))
            except ValueError:
                continue
        return str(p)

    def _iter_files(self, base: Path, glob: str):
        items = [base] if base.is_file() else sorted(base.rglob(glob))
        for f in items:
            if any(part in self._SKIP_DIR_PARTS for part in f.parts):
                continue
            if self._resolve(str(f), write=False) is None:
                continue
            yield f

    MODEL_FILE = run_records.MODEL_FILE

    def _is_model(self, p: Path) -> bool:
        return p.parent == self.agent.workdir and p.name == self.MODEL_FILE

    def _is_protected(self, p: Path) -> bool:
        return run_records.is_record(self.agent.workdir, p)

    def _install_model(self, p: Path) -> str:
        ex = self.agent.executor
        try:
            code = p.read_text(encoding="utf-8")
        except Exception as e:
            return f" NOTE: cannot read {self._display(p)} ({type(e).__name__}: {e})."
        if ex is not None:
            from sandbox.world_proxy import WorldModelProxy
            ld = ex.world_load(code)
            if not ld.get("loaded"):
                return (f" NOTE: it did NOT install as the world model ({ld.get('error')}); "
                        "the previously installed model is still active — fix and re-save.")
            self.agent.code = code
            self.agent.world = WorldModelProxy(ex, stateful=ld["stateful"],
                                               has_is_goal=bool(ld.get("has_is_goal")),
                                               has_win_condition=bool(ld.get("has_win_condition")))
            self.agent.save_code()
            self.agent.world.set_entry_grid(self.agent.entry_grids.get(self.agent._current_level),
                                            self.agent._current_level)
            kind = _kind_label(ld["stateful"])
            return (f" Installed as the live world model [{kind}];"
                    + (self._win_cond_note() if ld.get("has_win_condition") else "")
                    + self._planning_notes(bool(ld.get("has_is_goal")), bool(ld.get("has_heuristic")))
                    + self._backtest_hint())
        try:
            world = CodeWorldModel(code)
        except Exception as e:
            return (f" NOTE: it did NOT install as the world model ({type(e).__name__}: {e}); "
                    "the previously installed model is still active — fix and re-save.")
        self.agent.code = code
        self.agent.world = world
        self.agent.save_code()
        world.set_entry_grid(self.agent.entry_grids.get(self.agent._current_level),
                             self.agent._current_level)
        kind = _kind_label(world.stateful)
        return (f" Installed as the live world model [{kind}];"
                + (self._win_cond_note() if world.has_win_condition else "")
                + self._planning_notes(world.has_goal_pred, world.has_heuristic)
                + self._backtest_hint())

    @staticmethod
    def _planning_notes(has_goal_pred: bool, has_heuristic: bool) -> str:
        return ((" goal predicate defined (BFS goal search enabled)." if has_goal_pred
                 else " no goal predicate (BFS stops only on flag-predicted level_ups).")
                + (" bfs_heuristic defined (run_bfs uses A*)." if has_heuristic else ""))

    @staticmethod
    def _win_cond_note() -> str:
        return " is_win_condition defined (backtest-checked)."

    @staticmethod
    def _backtest_hint() -> str:
        return " Run backtest to check it against history."

    def tool_read_file(self, args: dict) -> str:
        p = self._resolve(args.get("path"), write=False)
        if p is None:
            return "ERROR: path not readable (allowed: your workdir + the world_model source)."
        if not p.exists():
            return f"ERROR: no such file: {args.get('path')}"
        if p.is_dir():
            return f"ERROR: {self._display(p)} is a directory — use find."
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"ERROR: cannot read: {type(e).__name__}: {e}"
        lines = text.splitlines()
        total = len(lines)
        offset = max(1, int(args.get("offset", 1) or 1))
        limit = int(args.get("limit", 2000) or 2000)
        start = min(offset - 1, total)
        end = min(start + limit, total)
        body, size = [], 0
        for i in range(start, end):
            row = f"{i + 1}\t{lines[i]}"
            size += len(row) + 1
            if size > 50_000:
                end = i
                break
            body.append(row)
        head = f"{self._display(p)} ({total} lines"
        head += f", showing {start + 1}-{end})" if (start or end < total) else ")"
        tail = f"\n\n(capped — use offset={end + 1} to continue.)" if end < total else ""
        return head + ":\n" + "\n".join(body) + tail

    def tool_write_file(self, args: dict) -> str:
        if _is_noop_sink(_arg(args, "path", "file_path", "filename")):
            return "OK: no-op (/dev/null is a discard sink — nothing written)."
        content = _arg(args, "content", "text")
        if not isinstance(content, str):
            return "ERROR: 'content' (string) is required"
        p = self._resolve(_arg(args, "path", "file_path", "filename"), write=True)
        if p is None:
            return "ERROR: write not allowed — you may only write under your workdir."
        if self._is_protected(p):
            return f"ERROR: {self._display(p)} is a protected run record — read-only."
        if p.is_dir():
            return f"ERROR: {self._display(p)} is a directory."
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        except Exception as e:
            return f"ERROR: cannot write: {type(e).__name__}: {e}"
        suffix = self._install_model(p) if self._is_model(p) else ""
        return f"OK: wrote {len(content)} bytes to {self._display(p)}." + suffix

    def tool_edit_file(self, args: dict) -> str:
        if _is_noop_sink(_arg(args, "path", "file_path", "filename")):
            return "OK: no-op (/dev/null is a discard sink — nothing edited)."
        old = _arg(args, "old_string", "old_str", "oldText", "old")
        new = _arg(args, "new_string", "new_str", "newText", "new")
        if not isinstance(old, str) or not isinstance(new, str):
            return ("ERROR: need the text to replace and its replacement, both strings "
                    "(use old_string/new_string; aliases old_str/new_str also accepted).")
        p = self._resolve(_arg(args, "path", "file_path", "filename"), write=True)
        if p is None:
            return "ERROR: edit not allowed — you may only edit files under your workdir."
        if self._is_protected(p):
            return f"ERROR: {self._display(p)} is a protected run record — read-only."
        if not p.is_file():
            return f"ERROR: no such file under workdir: {args.get('path')}"
        try:
            text = p.read_text(encoding="utf-8")
        except Exception as e:
            return f"ERROR: cannot read: {type(e).__name__}: {e}"
        count = text.count(old)
        if count == 0:
            return "ERROR: old_string not found (it must match exactly, including whitespace)."
        replace_all = bool(args.get("replace_all", False))
        if count > 1 and not replace_all:
            return f"ERROR: old_string occurs {count} times — add context to make it unique, or set replace_all=true."
        try:
            p.write_text(text.replace(old, new), encoding="utf-8")
        except Exception as e:
            return f"ERROR: cannot write: {type(e).__name__}: {e}"
        suffix = self._install_model(p) if self._is_model(p) else ""
        return f"OK: replaced {count if replace_all else 1} occurrence(s) in {self._display(p)}." + suffix

    def tool_cp(self, args: dict) -> str:
        src = self._resolve(_arg(args, "src", "source", "from"), write=False)
        if src is None:
            return "ERROR: source not readable (allowed: your workdir + the framework source)."
        if not src.is_file():
            return f"ERROR: source is not a file: {_arg(args, 'src', 'source', 'from')}"
        dst = self._resolve(_arg(args, "dst", "dest", "destination", "to"), write=True)
        if dst is None:
            return "ERROR: destination must be under your workdir."
        if dst.is_dir():
            dst = dst / src.name
        if self._is_protected(dst):
            return f"ERROR: refusing to overwrite run-critical file {self._display(dst)}."
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        except Exception as e:
            return f"ERROR: cannot copy: {type(e).__name__}: {e}"
        suffix = self._install_model(dst) if self._is_model(dst) else ""
        return f"OK: copied {self._display(src)} → {self._display(dst)}." + suffix

    def tool_mv(self, args: dict) -> str:
        src = self._resolve(_arg(args, "src", "source", "from"), write=True)
        if src is None:
            return "ERROR: source must be under your workdir (the framework source is read-only)."
        if src == self.agent.workdir:
            return "ERROR: refusing to move the workdir root."
        if not src.exists():
            return f"ERROR: no such file: {_arg(args, 'src', 'source', 'from')}"
        if self._is_protected(src):
            return f"ERROR: refusing to move run-critical file {self._display(src)}."
        dst = self._resolve(_arg(args, "dst", "dest", "destination", "to"), write=True)
        if dst is None:
            return "ERROR: destination must be under your workdir."
        if dst.is_dir():
            dst = dst / src.name
        if self._is_protected(dst):
            return f"ERROR: refusing to overwrite run-critical file {self._display(dst)}."
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
        except Exception as e:
            return f"ERROR: cannot move: {type(e).__name__}: {e}"
        suffix = self._install_model(dst) if self._is_model(dst) else ""
        return f"OK: moved {self._display(src)} → {self._display(dst)}." + suffix

    def tool_rm(self, args: dict) -> str:
        p = self._resolve(_arg(args, "path", "file_path", "filename", "target"), write=True)
        if p is None:
            return "ERROR: can only delete files under your workdir."
        if p == self.agent.workdir:
            return "ERROR: refusing to delete the workdir root."
        if not p.exists():
            return f"ERROR: no such path: {_arg(args, 'path', 'file_path', 'filename', 'target')}"
        if self._is_protected(p):
            return f"ERROR: refusing to delete run-critical file {self._display(p)}."
        recursive = bool(args.get("recursive", False))
        try:
            if p.is_dir():
                if not recursive:
                    return f"ERROR: {self._display(p)} is a directory — pass recursive=true to delete it."
                shutil.rmtree(p)
            else:
                p.unlink()
        except Exception as e:
            return f"ERROR: cannot delete: {type(e).__name__}: {e}"
        note = ""
        if self._is_model(p):
            note = (" NOTE: that was your world-model file; the in-memory model stays active "
                    "for now but will not survive a resume — re-create it before then.")
        return f"OK: deleted {self._display(p)}." + note

    def tool_grep(self, args: dict) -> str:
        pattern = args.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            return "ERROR: 'pattern' (regex) is required"
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return f"ERROR: bad regex: {e}"
        base = self._resolve(args.get("path") or ".", write=False)
        if base is None:
            return "ERROR: path not readable (allowed: your workdir + the world_model source)."
        glob = args.get("glob") or "*"
        limit = 100
        out: list[str] = []
        for f in self._iter_files(base, glob):
            if not f.is_file():
                continue
            try:
                if f.stat().st_size > 2_000_000:
                    continue
                content = f.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            for i, line in enumerate(content.splitlines(), 1):
                if rx.search(line):
                    out.append(f"{self._display(f)}:{i}: {line.strip()[:200]}")
                    if len(out) >= limit:
                        return "\n".join(out) + f"\n(stopped at {limit} matches.)"
        return "\n".join(out) if out else "(no matches)"

    def tool_find(self, args: dict) -> str:
        base = self._resolve(args.get("path") or ".", write=False)
        if base is None:
            return "ERROR: path not readable (allowed: your workdir + the world_model source)."
        if base.is_file():
            return self._display(base)
        pattern = args.get("pattern") or "*"
        out: list[str] = []
        for f in self._iter_files(base, pattern):
            out.append(self._display(f) + ("/" if f.is_dir() else ""))
            if len(out) >= 500:
                out.append("(truncated at 500 entries.)")
                break
        return "\n".join(out) if out else "(no matches)"

    _EXEC_OUTPUT_CAP = 30_000

    @staticmethod
    def _exec_env() -> dict:
        env = dict(os.environ)
        bindir = str(Path(sys.executable).resolve().parent)
        env["PATH"] = bindir + os.pathsep + env.get("PATH", "")
        env["PYTHONPATH"] = ""
        env["PYTHONIOENCODING"] = "utf-8"
        return env

    def _clip(self, text: str) -> str:
        cap = self._EXEC_OUTPUT_CAP
        if text is None:
            return ""
        if len(text) <= cap:
            return text
        head, tail = text[: cap * 2 // 3], text[-cap // 3 :]
        return f"{head}\n… [clipped {len(text) - cap} chars] …\n{tail}"

    def _fmt_proc(self, header: str, code: int, dt: float, out: str, err: str) -> str:
        parts = [f"{header}\nexit={code} in {dt:.2f}s"]
        out, err = out or "", err or ""
        if out.strip():
            parts.append("--- stdout ---\n" + self._clip(out))
        if err.strip():
            parts.append("--- stderr ---\n" + self._clip(err))
        if not out.strip() and not err.strip():
            parts.append("(no output)")
        return "\n".join(parts)

    _INSPECT_CAP = 400

    def _grid_from_source(self, args: dict, *, default_which: str) -> "tuple":
        src = str(args.get("source", "current") or "current").strip().lower()
        timeline = self.agent.timeline
        if src == "current":
            g = timeline[-1].after if timeline else self.agent.entry_grids.get(self.agent._current_level)
            return g, "current frame", None
        if src == "entry":
            lv = args.get("level", self.agent._current_level)
            try:
                lv = int(lv)
            except (TypeError, ValueError):
                return None, "", f"ERROR: level must be an int, got {args.get('level')!r}"
            g = self.agent.entry_grids.get(lv)
            if g is None:
                return None, "", f"ERROR: no entry grid recorded for level {lv}"
            return g, f"entry grid of level {lv}", None
        if src != "history":
            return None, "", f"ERROR: source must be 'current', 'history' or 'entry', got {src!r}"
        if not timeline:
            return None, "", "ERROR: history is empty."
        idx = args.get("index")
        if idx is None:
            return None, "", "ERROR: source='history' needs `index` (negatives count from the end)."
        try:
            i = int(idx)
        except (TypeError, ValueError):
            return None, "", f"ERROR: index must be an int, got {idx!r}"
        n = len(timeline)
        if i < 0:
            i += n
        if not 0 <= i < n:
            return None, "", f"ERROR: index out of range (history has {n} transitions: 0..{n-1})."
        which = str(args.get("which", default_which) or default_which).strip().lower()
        if which not in ("before", "after"):
            return None, "", f"ERROR: which must be 'before' or 'after', got {which!r}"
        g = self._history_frame(i, which)
        if g is None:
            return None, "", (f"ERROR: transition {i} has no {which}-frame recorded "
                              f"(the first transition of a run has no before-frame).")
        ts = timeline[i]
        act = f"action {ts.action_id}" + (f" @({ts.x},{ts.y})" if ts.action_id == 6 else "")
        return g, f"history[{i}] {which} ({act}, level {ts.before_levels})", None

    def _history_frame(self, i: int, which: str):
        timeline = self.agent.timeline
        if which == "after":
            return timeline[i].after
        return timeline[i - 1].after if i >= 1 else None

    @staticmethod
    def _crop(g, region):
        a = np.asarray(g)
        if region is None:
            return a, (0, 0), None
        if not (isinstance(region, (list, tuple)) and len(region) == 4):
            return None, (0, 0), "ERROR: region must be [x0, y0, x1, y1]."
        try:
            x0, y0, x1, y1 = (int(v) for v in region)
        except (TypeError, ValueError):
            return None, (0, 0), "ERROR: region values must be ints."
        H, W = a.shape
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W - 1, x1), min(H - 1, y1)
        if x0 > x1 or y0 > y1:
            return None, (0, 0), "ERROR: empty region after clamping to the grid."
        return a[y0:y1 + 1, x0:x1 + 1], (x0, y0), None

    @staticmethod
    def _render_with_rulers(a, origin=(0, 0)) -> str:
        a = np.asarray(a)
        x0, y0 = origin
        H, W = a.shape
        gut = max(3, len(str(y0 + H - 1)))
        tens = "".join(str(((x0 + j) // 10) % 10) if (x0 + j) % 10 == 0 else " " for j in range(W))
        ones = "".join(str((x0 + j) % 10) for j in range(W))
        head = [" " * (gut + 1) + tens, " " * (gut + 1) + ones]
        rows = [f"{y0 + i:>{gut}} {''.join(_HEX[int(v) & 0xF] for v in a[i])}" for i in range(H)]
        return "\n".join(head + rows)

    def tool_run_python(self, args: dict) -> str:
        ex = self.agent.executor
        code = _arg(args, "code", "source", "script")
        path = _arg(args, "path", "file", "file_path")
        timeout = self._exec_timeout(args)
        raw_argv = args.get("args")
        argv = ([str(a) for a in raw_argv] if isinstance(raw_argv, list)
                else ([str(raw_argv)] if raw_argv not in (None, "") else []))
        stdin = args.get("stdin")
        stdin = stdin if isinstance(stdin, str) else None
        if path:
            p = self._resolve(path, write=False)
            if p is None or not p.is_file():
                return f"ERROR: no such .py file under your workdir/framework: {path}"
            header = ("$ python " + self._display(p) + (" " + " ".join(argv) if argv else "")).rstrip()
        elif isinstance(code, str) and code.strip():
            header = "$ python -c <inline>" + (" " + " ".join(argv) if argv else "")
        else:
            return "ERROR: provide 'code' (inline Python) or 'path' (a .py file under your workdir)."
        if ex is not None:
            r = ex.run_python(code=(None if path else code), path=(str(p) if path else None),
                              args=argv, stdin=stdin, timeout=timeout)
            return self._fmt_from_result(header, r)
        py = sys.executable
        cmd = [py, str(p), *argv] if path else [py, "-c", code, *argv]
        return self._run_proc(cmd, env=self._exec_env(), header=header, timeout=timeout, stdin=stdin)

    def tool_run_shell(self, args: dict) -> str:
        ex = self.agent.executor
        command = _arg(args, "command", "cmd")
        if not isinstance(command, str) or not command.strip():
            return "ERROR: 'command' (a shell command string) is required."
        timeout = self._exec_timeout(args)
        cwd = self.agent.workdir
        raw_cwd = args.get("cwd")
        if raw_cwd:
            c = self._resolve(raw_cwd, write=False)
            if c is None or not c.is_dir():
                return f"ERROR: cwd must be an existing directory under your workdir/framework: {raw_cwd}"
            cwd = c
        if ex is not None:
            r = ex.run_shell(command, cwd=str(cwd), timeout=timeout)
            return self._fmt_from_result(f"$ {command}", r)
        shell = os.environ.get("SHELL") or "/bin/sh"
        return self._run_proc([shell, "-c", command], env=self._exec_env(),
                              header=f"$ {command}", timeout=timeout, cwd=cwd)

    def _sandbox_wrap(self, cmd: list[str]) -> list[str]:
        wd = str(Path(self.agent.workdir).resolve())
        src = str(Path(self.agent._src_dir).resolve())
        if _landlock_ok():
            return [sys.executable, str(_LANDLOCK_EXEC), str(_REPO_ROOT), wd, src, "--", *cmd]
        if _SANDBOX_EXEC:
            repo = str(_REPO_ROOT)
            if any('"' in s for s in (repo, wd, src)):
                return cmd
            profile = ("(version 1)(allow default)"
                       f'(deny file-read* (subpath "{repo}"))'
                       f'(deny file-write* (subpath "{repo}"))'
                       f'(allow file-read* (subpath "{wd}"))'
                       f'(allow file-write* (subpath "{wd}"))'
                       f'(allow file-read* (subpath "{src}"))')
            return [_SANDBOX_EXEC, "-p", profile, *cmd]
        if "auto" not in _SANDBOX_WARNED:
            _SANDBOX_WARNED.add("auto")
            print("WARNING: no exec sandbox backend available; run_python/run_shell are not "
                  "filesystem-confined.", file=sys.stderr)
        return cmd

    def _run_proc(self, cmd: list[str], *, env: dict, header: str,
                  timeout: float, cwd: Path | None = None, stdin: str | None = None) -> str:
        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                self._sandbox_wrap(cmd), cwd=str(cwd or self.agent.workdir), input=stdin,
                capture_output=True, text=True, timeout=timeout, env=env,
            )
        except FileNotFoundError as e:
            return f"ERROR: cannot execute: {e}"
        except subprocess.TimeoutExpired as e:
            dt = time.monotonic() - t0
            out = e.stdout if isinstance(e.stdout, str) else ""
            err = e.stderr if isinstance(e.stderr, str) else ""
            tail = self._fmt_proc(header, -1, dt, out, err)
            return (f"ERROR: timed out after {timeout:.0f}s — process killed. "
                    f"Partial output below.\n{tail}")
        dt = time.monotonic() - t0
        return self._fmt_proc(header, proc.returncode, dt, proc.stdout, proc.stderr)

    def _fmt_from_result(self, header: str, r: dict) -> str:
        rc = int(r.get("returncode", -1))
        dt = float(r.get("seconds", 0.0))
        out, err = r.get("stdout", ""), r.get("stderr", "")
        if r.get("timed_out"):
            tail = self._fmt_proc(header, rc, dt, out, err)
            return (f"ERROR: timed out after {float(r.get('timeout', 0)):.0f}s — process killed. "
                    f"Partial output below.\n{tail}")
        return self._fmt_proc(header, rc, dt, out, err)

    @staticmethod
    def _exec_timeout(args: dict, *, default: float = 60.0, cap: float = 300.0) -> float:
        try:
            t = float(args.get("timeout", default) or default)
        except (TypeError, ValueError):
            t = default
        return max(1.0, min(cap, t))

    def tool_inspect_grid(self, args: dict) -> str:
        mode = str(args.get("mode", "grid") or "grid").strip().lower()
        if mode == "diff":
            return self._inspect_diff(args)
        g, label, err = self._grid_from_source(args, default_which="after")
        if err:
            return err
        a, origin, err = self._crop(g, args.get("region"))
        if err:
            return err
        head = f"{label}: shape={np.asarray(g).shape[0]}x{np.asarray(g).shape[1]}"
        if args.get("region"):
            head += f", cropped to x{origin[0]}..{origin[0] + a.shape[1] - 1}, y{origin[1]}..{origin[1] + a.shape[0] - 1}"
        if mode == "counts":
            vals, cnts = np.unique(np.asarray(a), return_counts=True)
            body = ", ".join(f"{_HEX[int(v) & 0xF]}={int(c)}" for v, c in zip(vals, cnts))
            return f"{head}\ncell counts by value: {body}"
        if mode != "grid":
            return f"ERROR: mode must be 'grid', 'counts' or 'diff', got {mode!r}"
        return head + "\n" + self._render_with_rulers(a, origin)

    def _inspect_diff(self, args: dict) -> str:
        if str(args.get("source", "") or "").strip().lower() != "history":
            return "ERROR: mode='diff' needs source='history' (it diffs one transition's before vs after)."
        before, label, err = self._grid_from_source({**args, "which": "before"}, default_which="before")
        if err:
            return err
        after, _, err = self._grid_from_source({**args, "which": "after"}, default_which="after")
        if err:
            return err
        b, a = np.asarray(before), np.asarray(after)
        if b.shape != a.shape:
            return f"{label}: shape changed {b.shape} -> {a.shape} (whole board replaced)."
        ch = np.argwhere(b != a)
        if not len(ch):
            return f"{label.replace(' before', '')}: nothing changed (before == after)."
        lines = [f"{label.replace(' before', '')}: {len(ch)} cell(s) changed (x, y): old -> new"]
        for y, x in ch[:self._INSPECT_CAP]:
            lines.append(f"  ({int(x)},{int(y)}): {_HEX[int(b[y, x]) & 0xF]} -> {_HEX[int(a[y, x]) & 0xF]}")
        if len(ch) > self._INSPECT_CAP:
            lines.append(f"  … {len(ch) - self._INSPECT_CAP} more (crop with region= to narrow it down)")
        return "\n".join(lines)

    def tool_model_predict(self, args: dict) -> str:
        world = self.agent.world
        if world is None:
            return "ERROR: no world model installed yet — write world_model.py first."
        try:
            action = int(args.get("action"))
        except (TypeError, ValueError):
            return "ERROR: `action` (an int 0-7) is required."
        x, y = args.get("x"), args.get("y")
        if action == 6 and (x is None or y is None):
            return "ERROR: action 6 is a click — give x and y (0..63)."
        grid, label, err = self._grid_from_source({"source": "current"}, default_which="after")
        if err:
            return err

        cand_note = ""
        cand_path = _arg(args, "path", "file", "file_path")
        if cand_path:
            world, cand_note, err = self._candidate_world(cand_path)
            if err:
                return err

        want_state = bool(args.get("show_state"))
        try:
            out = self._predict_one(grid, action, x, y, world=world, want_state=want_state)
        finally:
            self._restore_live_model(bool(cand_path))
        if isinstance(out, str):
            return out
        pred, info, state_repr = out
        a, origin, err = self._crop(pred, args.get("region"))
        if err:
            return err
        flags = {k: bool(info.get(k)) for k in ("level_up", "dead", "win") if info.get(k)}
        act = f"action {action}" + (f" @({x},{y})" if action == 6 else "")
        head = (f"{cand_note or 'your model'}'s prediction for {act} from {label}"
                + (f"  flags: {flags}" if flags else "  flags: none"))
        tail = f"\nstate after this step: {state_repr}" if want_state else ""
        return head + "\n" + self._render_with_rulers(a, origin) + tail

    def _restore_live_model(self, was_candidate: bool) -> None:
        ex = self.agent.executor
        if not was_candidate or ex is None:
            return
        code = self.agent.code
        if not code:
            return
        try:
            ex.world_load(code)
        except Exception as e:
            print(f"[tools] WARNING: could not reinstall the live world model after a candidate "
                  f"model_predict ({type(e).__name__}: {e})", file=sys.stderr)

    def _candidate_world(self, path):
        p = self._resolve(path, write=False)
        if p is None or not p.is_file():
            return None, "", f"ERROR: no such .py file under your workdir: {path}"
        try:
            code = p.read_text(encoding="utf-8")
        except Exception as e:
            return None, "", f"ERROR: cannot read {self._display(p)}: {type(e).__name__}: {e}"
        ex = self.agent.executor
        note = f"candidate {self._display(p)}"
        if ex is not None:
            from sandbox.world_proxy import WorldModelProxy
            ld = ex.world_load(code)
            if not ld.get("loaded"):
                return None, "", (f"ERROR: {self._display(p)} did not compile ({ld.get('error')}); "
                                  "the live model is unchanged.")
            return WorldModelProxy(ex, stateful=bool(ld["stateful"]),
                                   has_is_goal=bool(ld.get("has_is_goal")),
                                   has_win_condition=bool(ld.get("has_win_condition"))), note, None
        try:
            return CodeWorldModel(code), note, None
        except Exception as e:
            return None, "", (f"ERROR: {self._display(p)} did not compile "
                              f"({type(e).__name__}: {e}); the live model is unchanged.")

    _STATE_REPR_CAP = 4000

    def _predict_one(self, grid, action, x, y, world=None, want_state=False):
        agent = self.agent
        world = world if world is not None else agent.world
        try:
            level = agent._current_level
            end = len(agent.timeline)
            if agent.executor is not None:
                r = world.predict_step(agent.timeline, level, agent.entry_grids, end,
                                       grid, action, x, y, want_state=want_state)
                if r is None:
                    return "ERROR: your model raised or timed out on that step."
                return r if want_state else (r[0], r[1], None)
            state = rollout_state(world, agent.timeline, agent.entry_grids, level, end)
            world.set_entry_grid(agent.entry_grids.get(level), level)
            pred, info, nxt = world.predict(state, grid, action, x, y)
            rep = None
            if want_state:
                try:
                    rep = repr(nxt)[:self._STATE_REPR_CAP]
                except Exception as e:
                    rep = f"<repr failed: {type(e).__name__}>"
            return np.asarray(pred), (info or {}), rep
        except Exception as e:
            return f"ERROR: your model raised {type(e).__name__}: {e}"

    def tool_run_bfs(self, args: dict) -> str:
        world = self.agent.world
        if world is None:
            return "ERROR: no world model installed; write world_model.py (write_file) first."
        max_depth = int(args.get("max_depth", 50) or 50)
        max_nodes = int(args.get("max_nodes", 20000) or 20000)
        target = str(args.get("target", "advance")).strip().lower()
        if target not in ("advance", "win", "level_up", "is_goal"):
            return f"ERROR: target must be one of advance/win/level_up/is_goal, got {target!r}"

        raw_actions = args.get("actions")
        if isinstance(raw_actions, list) and raw_actions:
            acts = [int(a) for a in raw_actions if int(a) not in (0, 6)]
        else:
            acts = [a for a in self.legal if a not in (0, 6)]
        clicks: list[tuple[int, int]] = []
        raw_clicks = args.get("clicks")
        if isinstance(raw_clicks, list):
            for c in raw_clicks:
                if isinstance(c, (list, tuple)) and len(c) == 2:
                    clicks.append((int(c[0]), int(c[1])))
        if not acts and not clicks:
            return "ERROR: nothing to search (no legal simple actions and no clicks given)."

        ex = self.agent.executor
        entry = self.agent.entry_grids.get(self.agent._current_level)
        allow_reset = bool(args.get("allow_reset", True))
        reset_used = allow_reset and entry is not None
        if ex is not None:
            r = world.bfs(self.grid, acts, clicks, target, self.agent.timeline,
                          self.agent.entry_grids, self.agent._current_level,
                          max_depth, max_nodes, allow_reset)
        else:
            world.set_entry_grid(entry, self.agent._current_level)
            start_state = self.agent._rollout(self.agent._current_level, len(self.agent.timeline))
            reset_grid = entry if reset_used else None
            reset_state = None
            if reset_grid is not None:
                try:
                    reset_state = world.init_state(entry)
                except Exception:
                    reset_grid = None
                    reset_used = False
            r = bfs(world, self.grid, acts, clicks=clicks, target=target, start_state=start_state,
                    reset_grid=reset_grid, reset_state=reset_state,
                    max_depth=max_depth, max_nodes=max_nodes)
        searched = (f"actions={acts}" + (f" + {len(clicks)} click(s)" if clicks else "")
                    + (" + RESET-first option" if reset_used else ""))
        return self._format_bfs_result(r, searched=searched, max_nodes=max_nodes, max_depth=max_depth)

    _ALGO_DESC = {
        "bfs": "BFS", "astar": "A* with your bfs_heuristic",
        "greedy": "greedy best-first fallback (ordered by your bfs_heuristic)",
        "bfs+novelty": "novelty-pruned BFS fallback",
    }

    @staticmethod
    def _plan_seq(path) -> list:
        return [({"action": a} if a == 0 or x is None else {"action": a, "x": x, "y": y})
                for (a, x, y) in path]

    @staticmethod
    def _speed_line(r: dict) -> str:
        ms = r.get("predict_ms")
        if not ms:
            return ""
        budget = float(r.get("time_budget_s") or BFS_SEARCH_S)
        return (f"Model speed: predict ≈ {ms:.2f} ms/call → ≈ {int(budget * 1000.0 / ms):,} nodes fit "
                f"in one {budget / 60:.0f}-min search.")

    def _format_bfs_result(self, r: dict, *, searched: str, max_nodes: int, max_depth: int) -> str:
        has_h = bool(r.get("has_heuristic"))
        phases = r.get("phases") or []
        ph = "; ".join(f"{p['algo']}: {p['expanded']} nodes/{p['secs']:.0f}s → {p['termination']}"
                       for p in phases)
        algo = self._ALGO_DESC.get(r.get("algo"), r.get("algo") or "search")
        lines: list[str] = []
        if r["plan"] is not None:
            seq = self._plan_seq(r["plan"])
            lines.append(
                f"BFS: goal in {len(r['plan'])} step(s) via {r['goal_reason']} [{algo}]; "
                f"expanded {r['expanded']} nodes, {r['distinct']} distinct states ({searched})."
            )
            if r.get("note"):
                lines.append(r["note"])
            if not r.get("optimal", True):
                lines.append(
                    "NOTE: the main search ran out of room before proving a shortest plan "
                    f"({ph}); this plan may be LONGER than optimal. "
                    + ("Define bfs_heuristic(grid[, state]) -> estimated actions remaining to enable "
                       "A* and get shorter plans." if not has_h else
                       "A sharper bfs_heuristic or a nearer waypoint usually shortens it.")
                )
            if r.get("heuristic_error"):
                lines.append(f"WARNING: bfs_heuristic raised {r['heuristic_error']} — the search "
                             "continued with h=0 (blind ordering) from that point.")
            if seq and seq[0].get("action") == 0:
                lines.append(
                    "NOTE: step 1 is RESET — it restarts the CURRENT level (back to its entry "
                    "layout, refunding level budgets); the rest of the plan runs from that clean "
                    "start. BFS chose this because it's the shortest path to the goal from here."
                )
            lines.append("(start already at goal — no actions needed.)" if not seq
                         else f"Plan (-> commit_actions): {seq}")
            lines.append("Predicted final grid:\n" + render_grid(r["final_grid"]))
            sp = self._speed_line(r)
            if sp:
                lines.append(sp)
            lines.append("Reminder: only as reliable as your model"
                         " — trust it only if backtest passed.")
            return "\n".join(lines)

        term = r["termination"]
        if term == "no_goal_capability":
            return ("BFS: cannot search — " + r["note"]
                    + " Define is_bfs_goal(grid) (or is_win_condition), or use target='advance' "
                      "to stop on level_up/win flags.")
        budget_s = float(r.get("time_budget_s") or BFS_SEARCH_S)
        stats = (f"expanded {r['expanded']} nodes, {r['distinct']} distinct states, deepest state "
                 f"reached: depth {r.get('depth_reached', 0)}; {ph}; {searched}")
        if term == "exhausted":
            lines.append(
                f"BFS: no goal found (exhausted — explored ALL {r['distinct']} states reachable under "
                f"{searched}; the model says the goal is UNREACHABLE from here). Rethink the "
                "model/goal, or add clicks / other actions. [" + stats + "]")
        elif term == "depth":
            lines.append(
                f"BFS: no goal found (depth — every path was truncated at max_depth={max_depth} "
                "before reaching a goal). Retry with a larger max_depth. [" + stats + "]")
        elif term == "timeout":
            lines.append(
                f"BFS: no goal found (timeout — the {budget_s / 60:.0f}-min wall-clock budget ran "
                "out). [" + stats + "]")
        elif term == "budget":
            lines.append(
                f"BFS: no goal found (budget — hit max_nodes={max_nodes}; {r['frontier']} states "
                "still pending). [" + stats + "]")
        else:
            lines.append(f"BFS: no goal found ({term}). [" + stats + "]")
        sp = self._speed_line(r)
        if sp:
            lines.append(sp)
        best = r.get("best")
        if has_h and best is not None and not r.get("heuristic_error"):
            h0 = r.get("h_start")
            h0s = "?" if h0 is None else f"{h0:g}"
            if best.get("depth", 0) == 0:
                lines.append(
                    f"Closest state by your bfs_heuristic: none better than the start (h={h0s}) — it "
                    "never decreased along any explored path, so it is probably measuring the wrong "
                    "thing, or the mechanic that lowers it is not modelled.")
            else:
                lines.append(
                    f"Closest state by your bfs_heuristic: h={best['h']:g} (start h={h0s}) at depth "
                    f"{best['depth']} via {self._plan_seq(best['path'])}:\n" + render_grid(best["grid"]))
        if r.get("heuristic_error"):
            lines.append(f"WARNING: bfs_heuristic raised {r['heuristic_error']} — the search "
                         "continued with h=0 (blind ordering) from that point; fix it first.")
        if term in ("timeout", "budget"):
            steps: list[str] = []
            if not has_h:
                steps.append(
                    "Define bfs_heuristic(grid[, state]) -> float: your estimate of the actions still "
                    "needed to reach the goal (0 once reached). run_bfs then switches to A* "
                    "automatically and, even when it fails, shows you the closest state it reached. "
                    "A rough estimate is fine — it can only cost efficiency, never correctness.")
                steps.append("Narrow the search: pass only the actions / click cells that can matter "
                             "here, or set a nearer waypoint (is_bfs_goal + target='is_goal').")
            else:
                steps.append(
                    "Sharpen bfs_heuristic so it falls steadily along a solution (see the closest "
                    "state above), or split the task with a nearer is_bfs_goal waypoint and "
                    "target='is_goal'.")
                steps.append("Narrow the search: pass only the actions / click cells that can matter here.")
            if term == "budget":
                if r.get("predict_ms"):
                    steps.append(
                        f"Raising max_nodes helps only up to the time budget: at this model's speed "
                        f"about {int(budget_s * 1000.0 / r['predict_ms']):,} nodes fit in "
                        f"{budget_s / 60:.0f} min.")
                else:
                    steps.append("Raising max_nodes helps only as far as the time budget allows.")
            else:
                steps.append("Raising max_nodes / max_depth will NOT help: the time budget is already "
                             "the binding cap.")
            lines.append("What works next, in order:")
            lines.extend(f"  {i}. {s}" for i, s in enumerate(steps, 1))
        return "\n".join(lines)
