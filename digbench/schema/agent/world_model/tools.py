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
from typing import Any, Optional


from . import run_records
from .secrets import scrub_env
from .world import CodeWorldModel, bfs, diff_summary

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LANDLOCK_EXEC = Path(__file__).resolve().parent / "landlock_exec.py"

_TOOL_BUDGET_S = 660.0

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


def _arg(args: dict, *keys: str):
    for k in keys:
        v = args.get(k)
        if v is not None:
            return v
    return None


def _is_noop_sink(path) -> bool:
    return isinstance(path, str) and path.strip() in ("/dev/null",)


def render_obs(obs: Optional[str]) -> str:
    if obs is None:
        return "(no observation)"
    obs = str(obs)
    n_lines = obs.count("\n") + 1
    block = f"⟪\n{obs}\n⟫" if "\n" in obs else f"⟪{obs}⟫"
    return f"{block}\n(repr: {obs!r}; {len(obs)} chars, {n_lines} line{'s' if n_lines != 1 else ''})"


def render_state_line(st) -> str:
    parts = [f"status={st.status}",
             f"level {st.level}/{st.max_level if st.max_level is not None else '?'}"]
    if st.lives_left is not None:
        parts.append(f"lives {st.lives_left}" + (f"/{st.starting_lives}" if st.starting_lives is not None else ""))
    if st.steps_remaining is not None:
        parts.append(f"steps_remaining {st.steps_remaining}"
                     + (f"/{st.max_steps}" if st.max_steps is not None else "")
                     + (" (creative budget)" if st.in_creative else " (this level)"))
    if st.mode is not None:
        parts.append(f"mode {st.mode}")
    return " | ".join(parts)


def render_creative_line(st) -> str:
    if st.creative_toggle is None and st.mode is None:
        return ""
    tog = st.creative_toggle
    avail = st.creative_toggle_available
    s = f"Creative mode: toggle action {tog!r}" if tog is not None else "Creative mode: present"
    if avail is not None:
        s += " (available now)" if avail else (
            " (NOT available now" + (f": {st.creative_unavailable_reason}" if st.creative_unavailable_reason else "") + ")")
    if st.in_creative:
        s += (" — you are IN creative mode: steps here do not count against the level budget and lives "
              "are safe, but level progress is paused; submit the toggle again to return to survival mode.")
    return s


def _read_notes(agent: Any, cap: int = 4000) -> str:
    try:
        text = (agent.workdir / "notes.md").read_text(encoding="utf-8").strip()
    except Exception:
        return "(no notes.md yet)"
    if not text:
        return "(notes.md is empty)"
    if len(text) > cap:
        text = text[:cap] + f"\n… (notes.md truncated at {cap} chars — prune it with edit_file)"
    return text


def _common_head(agent: Any, latest, legal: list[str]) -> list[str]:
    st = latest.state
    parts = []
    desc = getattr(agent, "description", None)
    if desc:
        parts.append(f"Task description (objective + special actions, NOT the rules): {desc}")
    parts.append(f"State: {render_state_line(st)}")
    cl = render_creative_line(st)
    if cl:
        parts.append(cl)
    if st.transition:
        parts.append(f"Transition just now: {st.transition}")
    parts.append(f"Legal actions now: {legal!r}")
    notes = getattr(agent, "harness_notes", None)
    if callable(notes):
        try:
            parts += list(notes(latest, legal))
        except Exception as e:
            parts.append(f"(harness notes unavailable: {type(e).__name__}: {e})")
    return parts


def render_situation(agent: Any, latest, legal: list[str]) -> str:
    parts = _common_head(agent, latest, legal)
    parts += [
        f"World model: {'installed' if agent.world is not None else 'NONE yet'}; "
        f"history: {len(agent.timeline)} transitions; current segment {agent._segment}.",
        f"Files: workdir (read/write) = {agent.workdir}; "
        f"framework source (read-only) = {agent._src_dir}.",
    ]
    if getattr(agent, "last_outcome", ""):
        parts.append(f"Last turn: {agent.last_outcome}")
    if getattr(agent, "last_suggestion", ""):
        parts.append(f"Your note-to-self from last turn (reconsider, don't just obey): {agent.last_suggestion}")
    if getattr(agent, "last_surprise", ""):
        parts.append(f"NOTE: {agent.last_surprise}")
    parts += [
        "",
        "Your notes (notes.md — maintain it with write_file/edit_file; keep it concise):",
        _read_notes(agent),
        "",
        "Current observation:",
        render_obs(latest.state.observation),
        "",
        "Decide the next action(s). Update your world model / notes, run a backtest "
        "or BFS as needed, then end by calling commit_actions.",
    ]
    return "\n".join(parts)


def render_observation(agent: Any, latest, legal: list[str]) -> str:
    parts = []
    if getattr(agent, "last_outcome", ""):
        parts.append(f"Result of your last commit: {agent.last_outcome}")
    if getattr(agent, "last_surprise", ""):
        parts.append(f"NOTE: {agent.last_surprise}")
    parts += _common_head(agent, latest, legal)
    parts += [
        f"World model: {'installed' if agent.world is not None else 'NONE yet'}; "
        f"history: {len(agent.timeline)} transitions; current segment {agent._segment}.",
        "",
        "Current observation:",
        render_obs(latest.state.observation),
        "",
        "Decide the next action(s) (update model/notes, backtest or BFS as needed), then "
        "end by calling commit_actions. If your memory of a rule/layout is fuzzy after a long "
        "session, re-read notes.md / world_model.py / read_history before deciding.",
    ]
    return "\n".join(parts)


def observation_content(agent: Any, latest, legal: list[str], *, continuing: bool) -> list[dict]:
    render = render_observation if continuing else render_situation
    return [{"type": "text", "text": render(agent, latest, legal)}]


SYSTEM_PROMPT = """\
You are an agent playing an unknown TEXT-BASED game (a DiG-bench game). We are not going to tell \
you the rules of this game — you have to figure them out for yourself, through interaction and \
experimentation.

GAME PROTOCOL (what the bench guarantees; everything else must be inferred from play):
- Levels, lives and steps. The aim is to reach as high a level as possible; completing every level
  wins the game. You advance levels by reaching certain states within the game — you will have to
  figure out what these are. Within each level you have a limited number of steps
  (`steps_remaining`); if you run out of steps you lose a life. It is also possible to lose a life
  by reaching certain states within the game. Losing a life restarts the CURRENT level with a fresh
  step budget; the harness tells you after every loss whether the restart board was identical to
  the level's first entry board. If you lose all your lives, the game is over.
- Scoring. You are scored on LEVELS BEATEN; step efficiency is NOT scored.
- Creative mode. Some games have a creative mode. You will know it exists when the state carries a
  `mode` field and a toggle action (`creative_toggle`, typically "/") appears among the legal
  actions. Submitting the toggle switches you into a sandbox where you can experiment safely
  without losing steps or lives (steps taken there do not count against the level budget; there is
  a large separate creative budget). Submit the toggle again to return to survival mode (whether
  the survival board is exactly as you left it is for you to verify — the task description may
  say). Level progress happens only in survival mode. It may be necessary to use creative mode to
  discover the rules without running out of steps. Some games make creative mode a CHALLENGE
  board: solving it (bench event `creative_solved`) or failing it (`creative_failed`) returns you
  to survival mode at NO cost; its final board is scored by run_backtest like a real completion.
- Observation: a text string — the rendered screen. It is shown to you VERBATIM between ⟪ ⟫
  markers and as a Python repr, so every character and every space is exact. Some games EMBED
  HISTORY in the observation (lines tagged `(previous)` and `(current)` with the action in
  between): only the last block — after the final blank line — is the current state. The harness
  scores your predictions on that current frame and reports a prefix-only difference as a
  warning, not a misprediction; reproduce the format, but never let the prefix drive your model.
- Actions: string tokens. Every state lists its legal actions; you may only submit one of those.
- A short TASK DESCRIPTION (objective and any special actions, NOT the rules) is given at the
  start. Level completion / level failure are announced through a `transition` message (which
  quotes the FINAL board of the level), the bench's events (also new actions being unlocked) and
  the counters (level, lives, steps_remaining).

Your method (learn the rules, then exploit them):
1. Build a WORLD MODEL as Python code defining ONE of these two functions:

     def step(obs, action):
         # obs is the observation string, action the action token string. Return the predicted
         # next observation string, or (next_obs, info) where
         # info = {"level_up": bool, "life_lost": bool, "dead": bool, "win": bool}.

     def predict(state, obs, action):
         # Same, plus a custom `state` you define (any picklable value; whatever you returned
         # last step is what you receive this step). MUST return (next_obs, info, next_state).
         # You may also define init_state(entry_obs) to provide the initial value — it is
         # called at every SEGMENT start (default {} if omitted).

   Define `predict` OR `step`, not both. SEGMENTS: the run is cut at every boundary step — a
   level clear, a level restart after a lost life, and a creative/survival mode switch. The
   framework rolls your state forward along the REAL history segment by segment (re-initializing
   at each segment start from that segment's entry observation), so you never store anything to
   disk — you only describe the transition. run_backtest and run_bfs work identically for both
   contracts (they thread state for you).

   GOAL PREDICATES — both optional; either form works: f(obs) or f(state, obs), return bool.
     def is_win_condition(...): your model of the LEVEL-COMPLETION RULE — True exactly on states
       whose entry completes the level. It is PART OF THE WORLD MODEL and run_backtest CHECKS it
       (mismatch kind win_cond): it must be False on every recorded state that did not complete,
       and True on your own predicted completion observations (coherent with your level_up flags —
       cleanest is to derive the flag from it inside step()/predict()). Encode your win-rule
       hypothesis here instead of leaving it implicit; run_bfs target='advance' stops on it.
     def is_bfs_goal(...): an OPTIONAL self-chosen waypoint ("get the key first") for run_bfs
       target='is_goal' — a planning instrument, never verified, defaults to is_win_condition.
   numpy is preloaded as `np`. A global `ENTRY_OBS` is also preloaded: the observation as the
   current segment FIRST appeared (the framework sets it per segment, and swaps it per segment
   during run_backtest). Ground this level's layout from ENTRY_OBS instead of hard-coding
   positions. Globals `CURRENT_LEVEL` (1-based level number, None if unknown) and `CURRENT_MODE`
   ("survival" / "creative" / None) are set alongside it — read them for per-level / per-mode
   branching, or record them in your state inside init_state. You may only import: numpy, math,
   collections, itertools, functools, heapq, re, string, copy, dataclasses, typing, enum,
   operator, bisect. No file/network/os access (history reaches your model only via the
   predict() arguments — never try to read files).
   FLAGS: return level_up / life_lost / dead / win in info whenever an action completes a level
   (on the last level also win), or loses a life by reaching a losing STATE (dead when it was the
   last life) — run_backtest checks those flags on every step. Losing a life because the level's
   STEP BUDGET ran out is a harness-level event: you are not required to predict it (the backtest
   is lenient there), but mind steps_remaining when you commit long plans. On boundary steps the
   state's observation is already the NEXT board (the next level's entry, the restarted level, the
   survival board), but the bench reports the FINAL board on which the level / challenge ended:
   your predicted next observation is scored against that final board, and is_win_condition must
   be True on the real board that completed a level (and False on the real board of a loss by
   state). That is the ONLY positive evidence a win rule can get — the turn header tells you on
   how many real completions yours has been confirmed. The creative toggle itself is a harness
   action: it is never scored and BFS never uses it.
2. Rewrite the code ONLY when reality surprised a prediction. Check consistency with
   run_backtest: a good model predicts every transition's next observation AND its
   level_up/life_lost/dead/win outcome exactly.
3. When run_backtest shows the model matches all history, TRUST it: use run_bfs to find a
   path to the goal, then commit that path. BFS prunes branches that lose a life and is capped at
   the current step budget. But backtest-green is NOT proven-correct — it only matches
   transitions you have actually walked.
4. When you are unsure, commit an exploratory action that will teach you the most about the
   rules.

Memory: this is ONE CONTINUOUS conversation that spans the whole game — you keep your own
reasoning and what you learned across turns, and each turn you are simply handed the NEXT
observation (what your last committed plan did, the new observation, counters and legal actions).
You do NOT restart from scratch each turn. BUT as the conversation grows it is AUTO-COMPACTED: the
system first drops older tool outputs, then summarizes older turns — so exact details from far
back (full observations, long tool dumps, precise numbers) may be lost or condensed. Therefore
keep your DURABLE memory in FILES, which survive compaction verbatim: your world-model code +
notes.md, plus the recorded ground-truth history (read_history). After a long stretch, or whenever
your memory of a rule/layout feels fuzzy, RE-READ notes.md / world_model.py / read_history
instead of trusting recollection. Your world model lives in world_model.py in your workdir:
read it with read_file, and writing or editing that file (write_file / edit_file) automatically
(re)compiles and installs it as the live model — there is no separate code tool, and you can edit
it incrementally (e.g. flip one flag) instead of resubmitting the whole source. If an edit fails
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
the workdir, and run-critical files (events.jsonl, run.json) are protected. A starter
random_agent.py and your notes.md are seeded in your workdir.
"""


_SYSTEM_PROMPT_TAIL = """
You MUST end every turn by calling commit_actions with one or more actions to execute. \
Committing is the only way to act on the game; everything else is deliberation.

STRONGLY RECOMMENDED: before you commit_actions, call run_backtest to check that your \
current code reproduces the previously recorded transitions. If it does not yet match the \
history, fix the model first — committing (especially a multi-step plan) on top of a model \
that can't even replay the past is unreliable.
"""


_EXEC_TOOLS_SYSTEM_NOTE = """
Code execution: you also have run_python and run_shell — a REAL Python/shell scratchpad running
in the project's conda env (numpy and the project deps available; the working dir is your
workdir). This is SEPARATE from your world model:
world_model.py still executes in the restricted sandbox (no os/file/network), whereas
run_python and run_shell are unrestricted real processes. Use them to prototype and UNIT-TEST a
transition rule on concrete strings before you encode it into step()/predict(), to analyse an
observation or your recorded events.jsonl, or to crunch history — then fold what you confirm back
into the world model and verify with run_backtest. They do NOT share memory with the live game (no current
observation/timeline in-process; reach data via files like events.jsonl) and cannot act on the
game — only commit_actions does that. The GAME SERVER is OFF-LIMITS to these tools: do not try to
reach the bench API, open sessions, or look for API tokens — the only way to act on the game is
commit_actions, and the rules must be inferred from interaction.
"""


def build_system_prompt() -> str:
    return SYSTEM_PROMPT + _EXEC_TOOLS_SYSTEM_NOTE + _SYSTEM_PROMPT_TAIL


TOOL_SPECS: list[dict] = [
    {
        "name": "run_backtest",
        "description": (
            "Replay the world model (step() or predict()) over all recorded transitions and check "
            "its predictions. For a predict() model it rolls your state forward along the real "
            "history segment by segment (re-initialized at each level entry, level restart and "
            "mode switch). Three things are checked: (1) the next OBSERVATION string, exactly — on "
            "ordinary steps against the recorded next observation, on boundary steps "
            "(level_up/life_lost/creative challenge solved or failed) against the FINAL board the "
            "bench reported for that level/challenge (a difference confined to an embedded "
            "history prefix is only a warning); (2) the level_up/life_lost/dead/win FLAGS from the "
            "model's info, on EVERY step — so the model must correctly predict which action "
            "completes a level or loses a life (budget-exhaustion losses are exempt; predicting "
            "the completion/loss of a creative challenge is accepted); (3) is_win_condition, if "
            "defined (kind win_cond): False on every recorded state that did not complete, True on "
            "the REAL final board of every level clear / solved challenge (this is what 'win rule "
            "confirmed on N real completions' counts), False on the real final board of a loss by "
            "state. Invalid (rejected) actions and the creative toggle are skipped. Output has "
            "three parts: an overall count, the full list of mismatched indices each tagged with "
            "the error kind (obs / level_up / life_lost / dead / win / win_cond / error), and full "
            "detail (observations + the model's wrong prediction) for the most-recent N "
            "mismatches. ENTRY_OBS / CURRENT_LEVEL / CURRENT_MODE "
            "are swapped to each transition's own segment while replaying, so a general model "
            "grounds correctly per level. By default all transitions are checked — PREFER "
            "backtesting against ALL levels: a regression on a past level means an edit broke a "
            "confirmed mechanism. You can still scope to localise a bug: a contiguous range "
            "(start..end), explicit indices, a single level (level=N or 'current'), a mode "
            "(survival/creative) and/or a segment. Use before trusting run_bfs. By default the LIVE "
            "world model is tested; pass `path` to instead test a CANDIDATE .py file under your "
            "workdir WITHOUT installing it — so you can keep several model variants as separate "
            "files and compare them rather than repeatedly overwriting world_model.py. The "
            "candidate is never installed; a compile error reports back and leaves the live model "
            "untouched."
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
                    "description": "how many of the most-recent mismatches to show in full (observations + prediction); default 5, 0 to suppress",
                },
                "start": {"type": "integer", "description": "scope: range start index, inclusive (negatives count from end). Defaults to 0 if only end given."},
                "end": {"type": "integer", "description": "scope: range end index, inclusive (negatives count from end). Defaults to last if only start given."},
                "indices": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "scope: backtest only these transition indices (negatives allowed); takes precedence over range",
                },
                "level": {"description": "scope: only transitions taken on this level — an int level number, or 'current' for the level being played now"},
                "mode": {"type": "string", "description": "scope: only transitions taken in this mode ('survival' or 'creative')"},
                "segment": {"type": "integer", "description": "scope: only transitions of this segment id"},
            },
        },
    },
    {
        "name": "read_history",
        "description": (
            "Inspect/search recorded real transitions (TimeSteps; ground truth: before "
            "observation / action / after observation, with counters, flags and the bench's own "
            "events). The output ALWAYS starts with a Summary line of aggregate metadata over the "
            "WHOLE history (level_ups, lives lost, deaths, wins, mode switches, invalid actions, "
            "per-action counts, max level, segments) — read it to answer 'how many level-ups/...'.\n"
            "Select WHICH transitions: most-recent N (limit), a contiguous range (start..end, "
            "inclusive), or explicit indices (negatives count from end, -1 = last). "
            "Filter by metadata with action / flags / status / level / mode / segment (combined "
            "with AND); filters apply across the whole history unless a range/indices is also "
            "given. detail='full' (DEFAULT) renders before/after observations verbatim (+ repr); "
            "'brief' gives one summary line per step (action, chars changed, counters, flags) "
            "with no observations. In a contiguous full view a step's before observation is "
            "omitted when it equals the previous step's after (the chain link)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "cap on how many most-recent (matching) steps to show when no range/indices given (default 10)"},
                "start": {"type": "integer", "description": "range start index, inclusive (negatives count from end). Defaults to 0 if only end given."},
                "end": {"type": "integer", "description": "range end index, inclusive (negatives count from end). Defaults to the last if only start given."},
                "indices": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "explicit transition indices to show (negatives allowed); takes precedence over range/limit",
                },
                "action": {"description": "filter: keep only steps whose action token matches this string or any string in this list"},
                "flags": {"description": "filter: keep only steps where ALL listed flags are true; allowed 'level_up','life_lost','dead','win','mode_switch','invalid','terminal' (string or list)"},
                "status": {"description": "filter: keep only steps whose resulting status matches; 'in_progress','completed','game_over' (string or list)"},
                "level": {"description": "filter: keep only steps taken on this level (int) or 'current'"},
                "mode": {"type": "string", "description": "filter: keep only steps taken in this mode ('survival'/'creative')"},
                "segment": {"type": "integer", "description": "filter: keep only steps of this segment id"},
                "detail": {
                    "type": "string",
                    "enum": ["brief", "full"],
                    "description": "'full' = before/after observations (DEFAULT); 'brief' = summaries only",
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
            "Breadth-first search INSIDE your world model for a shortest action sequence to a goal "
            "(reliable only if run_backtest passed, and only as good as your win rule — the output "
            "says on how many real completions it has been confirmed). A goal is a win, a level-up (advancing one "
            "level — search stops there, since the next board auto-switches and the model can't "
            "predict it), a state satisfying your is_win_condition, or (target='is_goal') your "
            "is_bfs_goal waypoint. Branches whose info says life_lost/dead are pruned. The search "
            "branches on the CURRENT legal actions (minus the creative toggle) unless you pass "
            "`actions`; in survival mode the plan length is capped at steps_remaining (a longer "
            "plan would lose a life before finishing). On success returns a commit_actions-ready "
            "plan plus the predicted final observation; on failure it says WHY: 'exhausted' (goal "
            "unreachable in the model), 'budget' (raise max_nodes), 'depth' (raise max_depth), or "
            "'timeout' (the search is capped at 10 minutes of wall clock)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "max_depth": {"type": "integer", "description": "max plan length (default 20; capped at steps_remaining in survival mode)"},
                "max_nodes": {"type": "integer", "description": "max nodes to expand (default 20000)"},
                "actions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "restrict the search to these action tokens (default: the current legal actions minus the creative toggle). Actions that are not legal NOW are still allowed — legality can change mid-plan and your model decides",
                },
                "target": {
                    "type": "string",
                    "enum": ["advance", "win", "level_up", "is_goal"],
                    "description": "what counts as the goal: 'advance' (default: win OR level_up OR is_win_condition), 'win' (full game win only), 'level_up' (next level only), 'is_goal' (your waypoint: is_bfs_goal, falling back to is_win_condition / legacy is_goal)",
                },
            },
        },
    },
    {
        "name": "run_python",
        "description": (
            "Run real Python in the project's conda env (numpy and the project deps available) and "
            "return its combined stdout/stderr + exit code. This is a GENERAL scratchpad "
            "subprocess — full standard library and filesystem — and is SEPARATE from your "
            "sandboxed world-model code (world_model.py still runs in the restricted namespace; "
            "this does not). `import numpy as np` etc. work (site-packages); the working directory "
            "is your workdir, so relative paths and files you wrote land there. The GAME SERVER is "
            "OFF-LIMITS: do not try to reach the bench API or look for tokens — infer the rules "
            "from interaction. Pass EITHER `code` (inline source, may be multi-line) OR `path` (a "
            ".py file under your workdir). Use it to: analyse observations or events.jsonl, "
            "prototype and unit-test a transition rule before encoding it into your world model, "
            "crunch history, or sanity-check a hypothesis. Output is capped (~30KB, head+tail). "
            "NOTE: it does NOT share memory with the live game — the current observation/timeline "
            "reach it only via files (e.g. read events.jsonl). It cannot act on the game; only "
            "commit_actions can."
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
            "Run a shell command with the project's conda env on PATH (so `python`/`pip` resolve "
            "to that interpreter) and return its combined stdout/stderr + exit code. The working "
            "directory defaults to your workdir. Use it for quick env/IO chores: list/inspect "
            "files, `python -c ...`, `pip show numpy`, run a script you wrote, etc. It is a real "
            "shell — prefer the dedicated read_file/write_file/grep/find tools for ordinary file "
            "work, and use this for things they can't do. The GAME SERVER is OFF-LIMITS: do not "
            "try to reach the bench API or look for tokens — infer the rules from interaction. "
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
    {
        "name": "commit_actions",
        "description": (
            "TERMINAL: end deliberation and execute these actions on the game, in order — they form "
            "a queue. After EACH action the world model is auto-checked against what actually "
            "happened: if it matches, the next queued action runs automatically without consulting "
            "you; the FIRST time reality diverges from the model (or whenever you have no model "
            "yet) the remaining actions are dropped and you are consulted again. Execution also "
            "stops at every boundary (level cleared, life lost, mode switch, game over/won) and if "
            "the next queued action is not legal at that point. So commit a long plan only when your "
            "model is accurate — otherwise just commit one exploratory action. Each action is a "
            "string token that must be legal when it is executed (the first one must be legal NOW). "
            "Each executed step is recorded as a TimeStep. GATES (the commit is REJECTED, nothing "
            "runs, and you are told why): a survival plan longer than steps_remaining; a plan your "
            "own installed model predicts loses a life (dry-run from the live state); on your LAST "
            "life, any predicted loss, any budget overrun, and any plan without a green full-scope "
            "run_backtest on the installed model; and toggle-only commits after several in a row. "
            "accept_risk=true overrides the budget / predicted-loss gates as a deliberate "
            "experiment (never on the last life) — then say in `reason` what it will teach you."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "accept_risk": {
                    "type": "boolean",
                    "description": "OPTIONAL: knowingly commit a plan the model predicts loses a life or that overruns the survival budget, as a deliberate experiment (ignored on the last life). Default false.",
                },
                "actions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"action": {"type": "string", "description": "one action token, e.g. '2' or '/'"}},
                        "required": ["action"],
                    },
                    "description": "the action queue, in execution order (plain strings are accepted too)",
                },
                "reason": {
                    "type": "string",
                    "description": "OPTIONAL: why you are committing THESE actions now (justification for this step). It is forwarded to the bench as the move's reasoning note.",
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
]


def tool_specs() -> list[dict]:
    return TOOL_SPECS


class ToolBox:

    def __init__(self, agent: Any, obs: Optional[str], legal: list[str]) -> None:
        self.agent = agent
        self.obs = obs
        self.legal = [str(a) for a in (legal or [])]

    _MODEL_EXEC_TOOLS = frozenset({"run_backtest", "run_bfs"})

    def dispatch(self, name: str, args: dict) -> str:
        fn = getattr(self, f"tool_{name}", None)
        if fn is None:
            return f"ERROR: unknown tool {name!r}"
        budget = _TOOL_BUDGET_S
        use_alarm = (name in self._MODEL_EXEC_TOOLS
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

    def tool_run_backtest(self, args: dict) -> str:
        cand_note = ""
        candidate_code = None
        raw_path = _arg(args, "path", "file", "file_path")
        if raw_path:
            p = self._resolve(raw_path, write=False)
            if p is None or not p.is_file():
                return f"ERROR: no such .py file under your workdir/framework: {raw_path}"
            candidate_code = p.read_text(encoding="utf-8")
            try:
                world = CodeWorldModel(candidate_code)
            except Exception as e:
                return (f"ERROR: candidate {self._display(p)} failed to compile/load "
                        f"({type(e).__name__}: {e}); not backtested, live model unchanged.")
            cand_note = (f"candidate file {self._display(p)} (NOT installed; "
                         f"{'predict' if world.stateful else 'step'}) — ")
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
                level = int(self.agent.latest.state.level)
                desc_parts.append(f"level {level} (current)")
            else:
                try:
                    level = int(lv)
                except (TypeError, ValueError):
                    return f"ERROR: level must be an int or 'current', got {lv!r}"
                desc_parts.append(f"level {level}")
            candidates = [i for i in candidates if timeline[i].level_before == level]
        if args.get("mode"):
            mode = str(args["mode"]).strip().lower()
            desc_parts.append(f"mode {mode}")
            candidates = [i for i in candidates if (timeline[i].mode_before or "survival").lower() == mode]
        if args.get("segment") is not None:
            try:
                seg = int(args["segment"])
            except (TypeError, ValueError):
                return f"ERROR: segment must be an int, got {args['segment']!r}"
            desc_parts.append(f"segment {seg}")
            candidates = [i for i in candidates if timeline[i].segment == seg]
        scope = " ∩ ".join(desc_parts) if desc_parts else "all transitions"

        results = self._backtest_rollout(world)
        total = ok = skipped = 0
        n_warn = 0
        win_verified = 0
        boundary_scored = 0
        mismatches: list[dict] = []
        breakdown: dict = {}
        for i in candidates:
            r = results.get(i)
            if r is None:
                skipped += 1
                continue
            ts = timeline[i]
            total += 1
            key = ((ts.mode_before or "survival"), int(ts.level_before))
            cell = breakdown.setdefault(key, [0, 0])
            cell[1] += 1
            n_warn += len(r.get("warnings") or [])
            if r.get("win_verified"):
                win_verified += 1
            if r.get("terminal") and r.get("final_frame") is not None:
                boundary_scored += 1
            if r["errors"]:
                mismatches.append({"i": i, "ts": ts, **r})
            else:
                ok += 1
                cell[0] += 1
        full_scope = not desc_parts
        if candidate_code is None and world is self.agent.world:
            note = getattr(self.agent, "note_backtest", None)
            if callable(note):
                note(ok=ok, total=total, mismatches=len(mismatches), win_verified=win_verified,
                     full_scope=full_scope, breakdown=breakdown)

        has_win = bool(getattr(world, "has_win_condition", False)) if world is not None else False

        skipped_note = f"; {skipped} skipped (invalid actions / plain mode toggles)" if skipped else ""
        if total == 0:
            lines = [f"backtest [{cand_note}{scope}]: nothing to check yet{skipped_note}. "
                     "Take a few actions first, or widen the selection."]
            return "\n".join(lines)
        if not mismatches:
            head = (f"backtest [{cand_note}{scope}] — GREEN: {ok}/{total} checked transitions reproduced "
                    f"exactly (0 mismatches{skipped_note}).")
        else:
            head = (f"backtest [{cand_note}{scope}] — RED: {ok}/{total} reproduced, "
                    f"{len(mismatches)} mismatch(es){skipped_note}.")
        lines = [head]

        cur_level = int(self.agent.latest.state.level)
        parts = []
        for (mode, level), (o, t) in sorted(breakdown.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            parts.append(f"{mode} L{level} {o}/{t}")
        ev = "Evidence by mode and level: " + " · ".join(parts) + "."
        surv_cur = breakdown.get(("survival", cur_level), [0, 0])[1]
        if self.agent.latest.state.mode is not None and surv_cur == 0 and full_scope:
            ev += (f" No survival-mode evidence on level {cur_level} yet — everything green here was "
                   "learned on creative boards.")
        lines.append(ev)
        extra = []
        if boundary_scored:
            extra.append(f"{boundary_scored} boundary step(s) scored against the bench's final board")
        if n_warn:
            extra.append(f"{n_warn} step(s) matched on the current frame only (history prefix differs; accepted)")
        if extra:
            lines.append("; ".join(extra) + ".")
        if has_win:
            if win_verified:
                lines.append(f"Win rule: is_win_condition confirmed on {win_verified} real completion(s) in scope.")
            else:
                lines.append("Win rule: is_win_condition NOT confirmed on any real completion in scope — a "
                             "hypothesis until a level clear (or a solved creative challenge) makes it True on "
                             "a real final board.")
        else:
            lines.append("Win rule: no is_win_condition defined — BFS can only stop on flag-predicted level_ups; "
                         "encode your win hypothesis so the backtest can check it.")

        if not mismatches:
            lines.append("Verdict: the model explains everything it has seen in scope — safe to plan with run_bfs "
                         "(remember a green backtest only covers moves you have actually walked).")
            return "\n".join(lines)

        def flag_words(info: dict, ts) -> str:
            p = [k for k in ("level_up", "life_lost", "dead", "win") if (info or {}).get(k)] or ["nothing special"]
            a = ts.flag_names() or ["nothing special"]
            return f"predicted {'+'.join(p)}, actually {'+'.join(a)}"

        def one_line(m: dict) -> str:
            ts = m["ts"]
            where = f"{ts.mode_before or 'survival'} L{ts.level_before}"
            what: list[str] = []
            kinds = m["kinds"]
            if "error" in kinds:
                what.append(next((e for e in m["errors"] if "raised" in e), "model raised"))
            if any(k in kinds for k in ("level_up", "life_lost", "dead", "win")):
                what.append("flags: " + flag_words(m["info"], ts))
            if "obs" in kinds:
                target = m.get("final_frame") if m["terminal"] else m["after"]
                what.append(("final board wrong: " if m["terminal"] else "next observation wrong: ")
                            + short_diff(m["pred"], target))
            if "win_cond" in kinds:
                msg = next((e for e in m["errors"] if "win_condition" in e), "win rule inconsistent")
                what.append(msg.replace("win_condition error: ", "win rule: ").replace(
                    "win_condition incoherence: ", "win rule: "))
            tag = ""
            if ts.level_up:
                tag = " [level clear]"
            elif ts.challenge_outcome:
                tag = f" [creative challenge {ts.challenge_outcome}]"
            elif ts.life_lost:
                tag = " [life lost]"
            return f"  #{m['i']}  {where}  action {ts.action!r}{tag}  → " + "; ".join(what)

        stateful = bool(getattr(world, "stateful", False)) if world is not None else False
        lines.append("Mismatches, oldest first"
                     + (" (stateful model: fix the EARLIEST one in each segment first — later ones may be "
                        "cascades of a wrong state)" if stateful else "")
                     + ":")
        shown = mismatches[:25]
        lines += [one_line(m) for m in shown]
        if len(mismatches) > len(shown):
            lines.append(f"  … {len(mismatches) - len(shown)} more (narrow the scope with level=/mode=/start..end to see them).")

        if max_details > 0:
            recent = mismatches[-max_details:]
            lines.append(f"\nThe {len(recent)} most recent mismatch(es) in full:")
            for m in recent:
                ts, i = m["ts"], m["i"]
                lines.append(f"\n#{i}  {ts.mode_before or 'survival'} L{ts.level_before}  action {ts.action!r}  "
                             f"segment {ts.segment}  actual flags={ts.flag_names() or ['none']}"
                             + (f"  creative challenge {ts.challenge_outcome}" if ts.challenge_outcome else ""))
                for e in m["errors"]:
                    lines.append(f"  ! {e}")
                lines.append("  before:\n" + _indent(render_obs(m["before"])))
                if m["terminal"] and m.get("final_frame") is not None:
                    lines.append("  actual FINAL board of the level/challenge (bench-reported):\n"
                                 + _indent(render_obs(m["final_frame"])))
                    lines.append("  (after it the state shows the next board:\n" + _indent(render_obs(m["after"])) + ")")
                elif m["terminal"]:
                    lines.append("  actual after (a new board — not scored on this step):\n" + _indent(render_obs(m["after"])))
                else:
                    lines.append("  actual after:\n" + _indent(render_obs(m["after"])))
                if m["pred"] is None:
                    lines.append("  predicted: (predict()/step() raised — no prediction)")
                else:
                    pflags = [k for k in ("level_up", "life_lost", "dead", "win") if m["info"].get(k)] or ["none"]
                    lines.append(f"  predicted flags={pflags}")
                    if not m["terminal"] or m.get("final_frame") is not None:
                        lines.append("  predicted after:\n" + _indent(render_obs(m["pred"])))
                        target = m.get("final_frame") if m["terminal"] else m["after"]
                        ad = aligned_diff(m["before"], target, m["pred"], pad="    ")
                        if ad:
                            lines.append("  aligned (carets mark where predicted ≠ actual):\n" + ad)
        lines.append("\nNext: fix the RULE behind the earliest mismatch (not that one transition), re-run "
                     "run_backtest, and only then plan.")
        return "\n".join(lines)

    def _backtest_rollout(self, world: Any = None) -> dict:
        from .world import backtest_rollout
        world = world if world is not None else self.agent.world
        return backtest_rollout(world, self.agent.timeline, self.agent.entries)

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
        full = detail != "brief"

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
            f"detail={'full' if full else 'brief'}:",
        ]
        if not matched:
            lines.append("(no transitions match)")
            return "\n".join(lines)

        prev_idx = None
        for i in matched:
            ts = timeline[i]
            b, a = ts.before_obs, ts.after
            if ts.invalid:
                diff_text = "INVALID action (rejected by the bench; state unchanged)"
            elif ts.terminal:
                diff_text = "boundary → new board (" + diff_summary(b, a).split(";")[0] + ")"
            else:
                diff_text = diff_summary(b, a)
            st = ts.state
            counters = (f"L{ts.level_before}→{ts.level_after}"
                        + (f" lives {ts.lives_before}→{ts.lives_after}" if ts.lives_after is not None else "")
                        + (f" steps {ts.steps_before}→{ts.steps_after}" if ts.steps_after is not None else "")
                        + (f" mode {ts.mode_before}→{ts.mode_after}" if ts.mode_after is not None else ""))
            evs = [str(e.get("type")) for e in ts.events if e.get("type")]
            lines.append(
                f"#{i} action={ts.action!r}; seg={ts.segment}; {counters}; {diff_text}; "
                f"status={st.status}; flags={ts.flag_names() or ['none']}"
                + (f"; events={evs}" if evs else "")
                + (f"; transition={st.transition!r}" if st.transition else "")
            )
            if full:
                if b is None:
                    lines.append("  before: (none)")
                elif prev_idx == i - 1:
                    lines.append(f"  before: == #{i - 1} after (omitted)")
                else:
                    lines.append("  before:\n" + _indent(render_obs(b)))
                lines.append("  after:\n" + _indent(render_obs(a)))
                if st.actions:
                    lines.append(f"  legal after: {st.actions!r}")
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
        acts = Counter(ts.action for ts in timeline)
        return (
            "Summary: "
            f"level_ups={sum(1 for ts in timeline if ts.level_up)} "
            f"lives_lost={sum(1 for ts in timeline if ts.life_lost)} "
            f"deaths={sum(1 for ts in timeline if ts.dead)} "
            f"wins={sum(1 for ts in timeline if ts.win)} "
            f"mode_switches={sum(1 for ts in timeline if ts.mode_switch)} "
            f"invalid={sum(1 for ts in timeline if ts.invalid)}; "
            f"by-action={{{', '.join(f'{a!r}:{acts[a]}' for a in sorted(acts))}}}; "
            f"max_level={max((ts.level_after for ts in timeline), default=0)}; "
            f"segments={max((ts.segment for ts in timeline), default=0) + 1}"
        )

    def _history_filters(self, args: dict) -> dict | None:
        f: dict = {}
        a = args.get("action")
        if a is not None:
            f["actions"] = {str(x) for x in (a if isinstance(a, list) else [a])}
        fl = args.get("flags")
        if fl:
            allowed = {"level_up", "life_lost", "dead", "win", "mode_switch", "invalid", "terminal"}
            got = [x for x in (fl if isinstance(fl, list) else [fl]) if x in allowed]
            if got:
                f["flags"] = got
        st = args.get("status")
        if st:
            f["statuses"] = {str(x).strip().lower() for x in (st if isinstance(st, list) else [st])}
        lv = args.get("level")
        if lv is not None:
            if isinstance(lv, str) and lv.strip().lower() == "current":
                f["level"] = int(self.agent.latest.state.level)
            else:
                try:
                    f["level"] = int(lv)
                except (TypeError, ValueError):
                    pass
        if args.get("mode"):
            f["mode"] = str(args["mode"]).strip().lower()
        if args.get("segment") is not None:
            try:
                f["segment"] = int(args["segment"])
            except (TypeError, ValueError):
                pass
        return f or None

    @staticmethod
    def _match_history(ts, filt: dict | None) -> bool:
        if not filt:
            return True
        if "actions" in filt and ts.action not in filt["actions"]:
            return False
        if "flags" in filt and not all(getattr(ts, fl) for fl in filt["flags"]):
            return False
        if "statuses" in filt and ts.status.lower() not in filt["statuses"]:
            return False
        if "level" in filt and ts.level_before != filt["level"]:
            return False
        if "mode" in filt and (ts.mode_before or "survival").lower() != filt["mode"]:
            return False
        if "segment" in filt and ts.segment != filt["segment"]:
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
        if "statuses" in filt:
            parts.append(f"status in {sorted(filt['statuses'])}")
        if "level" in filt:
            parts.append(f"level {filt['level']}")
        if "mode" in filt:
            parts.append(f"mode {filt['mode']}")
        if "segment" in filt:
            parts.append(f"segment {filt['segment']}")
        return ", ".join(parts)

    _SKIP_DIR_PARTS = frozenset({"__pycache__", ".git", ".mypy_cache", ".pytest_cache"})

    def _resolve(self, path: Any, *, write: bool) -> Path | None:
        if not isinstance(path, str) or not path.strip():
            return None
        roots = [self.agent.workdir] if write else [self.agent.workdir, self.agent._src_dir]
        try:
            p = Path(path)
            if not p.is_absolute():
                p = self.agent.workdir / p
            p = p.resolve()
        except Exception:
            return None
        return p if any(p == r or p.is_relative_to(r) for r in roots) else None

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
        try:
            code = p.read_text(encoding="utf-8")
        except Exception as e:
            return f" NOTE: cannot read {self._display(p)} ({type(e).__name__}: {e})."
        try:
            world = CodeWorldModel(code)
        except Exception as e:
            return (f" NOTE: it did NOT install as the world model ({type(e).__name__}: {e}); "
                    "the previously installed model is still active — fix and re-save.")
        self.agent.code = code
        self.agent.world = world
        self.agent.save_code()
        world.set_entry(self.agent.entries.get(self.agent._segment))
        kind = "stateful (predict)" if world.stateful else "stateless (step)"
        return (f" Installed as the live world model [{kind}];"
                + (" is_win_condition defined (backtest-checked)." if world.has_win_condition else "")
                + (" goal predicate defined (BFS goal search enabled)." if world.has_goal_pred
                   else " no goal predicate (BFS stops only on flag-predicted level_ups).")
                + " Run run_backtest to check it against history.")

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
    def _exec_timeout(args: dict, *, default: float = 60.0, cap: float = 300.0) -> float:
        try:
            t = float(args.get("timeout", default) or default)
        except (TypeError, ValueError):
            t = default
        return max(1.0, min(cap, t))

    @staticmethod
    def _exec_env() -> dict:
        env = scrub_env()
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

    def tool_run_python(self, args: dict) -> str:
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
        py = sys.executable
        cmd = [py, str(p), *argv] if path else [py, "-c", code, *argv]
        return self._run_proc(cmd, env=self._exec_env(), header=header, timeout=timeout, stdin=stdin)

    def tool_run_shell(self, args: dict) -> str:
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
        shell = os.environ.get("SHELL") or "/bin/sh"
        return self._run_proc([shell, "-c", command], env=self._exec_env(),
                              header=f"$ {command}", timeout=timeout, cwd=cwd)


    def tool_run_bfs(self, args: dict) -> str:
        world = self.agent.world
        if world is None:
            return "ERROR: no world model installed; write world_model.py (write_file) first."
        max_depth = int(args.get("max_depth", 20) or 20)
        max_nodes = int(args.get("max_nodes", 20000) or 20000)
        target = str(args.get("target", "advance")).strip().lower()
        if target not in ("advance", "win", "level_up", "is_goal"):
            return f"ERROR: target must be one of advance/win/level_up/is_goal, got {target!r}"

        st = self.agent.latest.state
        toggle = st.creative_toggle
        raw_actions = args.get("actions")
        if isinstance(raw_actions, list) and raw_actions:
            acts = [str(a) for a in raw_actions if str(a) != toggle]
        else:
            acts = [a for a in self.legal if a != toggle]
        if not acts:
            return "ERROR: nothing to search (no legal non-toggle actions and none given)."

        depth_note = ""
        if st.in_creative:
            depth_note = (" NOTE: you are in CREATIVE mode — this plan starts from the creative board; "
                          "level progress happens only in survival mode, so a plan found here must be "
                          "re-derived from the survival board (toggle back first, then run_bfs again) "
                          "before it can clear the level.")
        elif st.steps_remaining is not None and st.steps_remaining < max_depth:
            max_depth = max(1, int(st.steps_remaining))
            depth_note = (f" max_depth capped at steps_remaining={st.steps_remaining} (survival mode: a "
                          "longer plan loses a life before it finishes).")

        seg = self.agent._segment
        world.set_entry(self.agent.entries.get(seg))
        start_state = self.agent._rollout(seg, len(self.agent.timeline))
        r = bfs(world, self.obs, acts, target=target, start_state=start_state,
                max_depth=max_depth, max_nodes=max_nodes)
        searched = f"actions={acts!r}"

        if r["plan"] is not None:
            seq = [{"action": a} for a in r["plan"]]
            lines = [
                f"BFS: goal in {len(r['plan'])} step(s) via {r['goal_reason']}; "
                f"expanded {r['expanded']} nodes, {r['distinct']} distinct states ({searched}).{depth_note}"
            ]
            if r["note"]:
                lines.append(r["note"])
            lines.append("(start already at goal — no actions needed.)" if not seq
                         else f"Plan (-> commit_actions): {seq}")
            lines.append("Predicted final observation:\n" + _indent(render_obs(r["final_obs"])))
            bt = getattr(self.agent, "_last_backtest", None)
            verified = getattr(self.agent, "_model_verified", lambda: False)()
            if getattr(world, "has_win_condition", False):
                wv = (bt or {}).get("win_verified") if verified else None
                if wv:
                    lines.append(f"Win rule status: is_win_condition confirmed on {wv} real completion(s).")
                else:
                    lines.append("Win rule status: is_win_condition NOT confirmed on any real completion "
                                 "(or the installed model has no green backtest) — this plan is a bet on a "
                                 "hypothesis; test it in creative mode if you can.")
            lines.append("Reminder: only as reliable as your model — trust it only if run_backtest passed"
                         + ("" if verified else " (the installed model has NO green full-scope backtest right now)")
                         + ".")
            return "\n".join(lines)

        term = r["termination"]
        if term == "timeout":
            return ("BFS: timed out — only searches that finish within 10 minutes are supported. "
                    "Please reconsider your approach (e.g. fix the world model, or narrow the "
                    "action set) so the search completes faster.")
        if term == "no_goal_capability":
            return ("BFS: cannot search — " + r["note"]
                    + " Define is_bfs_goal(obs) (or is_win_condition), or use target='advance' "
                      "to stop on level_up/win flags.")
        why = {
            "exhausted": (f"exhausted — explored all {r['distinct']} reachable states under {searched}; "
                          "the model says the goal is UNREACHABLE from here (within the depth cap). "
                          "Rethink the model/goal, or widen the action set."),
            "budget": (f"budget — hit max_nodes ({max_nodes}); {r['frontier']} states still pending. "
                       "Retry with a larger max_nodes."),
            "depth": (f"depth — all paths truncated at max_depth ({max_depth}).{depth_note or ' Retry with a larger max_depth.'}"),
        }.get(term, term)
        return (f"BFS: no goal found ({why}) "
                f"[expanded {r['expanded']} nodes, {r['distinct']} distinct states, {searched}].")


def _indent(text: str, pad: str = "    ") -> str:
    return "\n".join(pad + line for line in str(text).split("\n"))


def short_diff(pred, actual) -> str:
    if pred is None or actual is None:
        return "no prediction" if pred is None else "no actual observation"
    if pred == actual:
        return "identical"
    if len(pred) != len(actual):
        n = min(len(pred), len(actual))
        first = next((i for i in range(n) if pred[i] != actual[i]), n)
        return (f"length {len(pred)} predicted vs {len(actual)} actual; first difference at index {first} "
                f"(predicted {pred[first:first + 6]!r} vs actual {actual[first:first + 6]!r})")
    idx = [i for i in range(len(actual)) if pred[i] != actual[i]]
    if len(idx) == 1:
        i = idx[0]
        return f"1 char differs at index {i}: predicted {pred[i]!r}, actual {actual[i]!r}"
    head = ", ".join(str(i) for i in idx[:6]) + (", …" if len(idx) > 6 else "")
    i = idx[0]
    return (f"{len(idx)} chars differ at indices {head}; at {i} predicted {pred[i]!r}, actual {actual[i]!r}")


def aligned_diff(before, actual, pred, pad: str = "    ") -> str:
    rows = [("before", before), ("actual", actual), ("predicted", pred)]
    if any(v is None or "\n" in str(v) for _, v in rows):
        return ""
    width = max(len(str(v)) for _, v in rows)
    a, p = str(actual), str(pred)
    carets = "".join("^" if (i >= len(a) or i >= len(p) or a[i] != p[i]) else " " for i in range(width)).rstrip()
    label = max(len(k) for k, _ in rows)
    out = [f"{pad}{k.ljust(label)}  {v}" for k, v in rows]
    out.append(f"{pad}{' ' * label}  {carets}")
    return "\n".join(out)
