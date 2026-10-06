from __future__ import annotations

import json
import logging
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from arcengine import FrameData, GameAction, GameState

from agent.agent import Agent

from .budgets import STEP_CHECK_S
from .claude_driver import ClaudeDriver
from .events import (
    ActionTaken,
    EventSink,
    ModelMispredicted,
    RunFinished,
    RunStarted,
    ToolFinished,
    ToolStarted,
    TurnCommitted,
    TurnFallback,
    TurnStarted,
    encode_grid,
)
from .stream_norm import make_on_event
from .timestep import TimeStep, decode_ticks, encode_ticks
from .tools import SYSTEM_PROMPT, TOOL_SPECS, ToolBox, observation_content
from .world import CodeWorldModel, rollout_state

logger = logging.getLogger(__name__)
_SRC_DIR = Path(__file__).resolve().parent


ActionStep = tuple[int, Optional[int], Optional[int]]

_NOTES_SEED = """\
# Notes — your living scratchpad (shown to you every turn).
# Keep it CONCISE; edit and PRUNE stale entries with write_file / edit_file as you learn.

## Action semantics (confirmed / guessed)
<!-- e.g. "confirmed: action 1 does X"; "guess: action 5 does Y" -->

## Current level
<!-- same vs previous levels; new motifs; goal hypothesis; current plan -->

## Hypotheses to test
<!-- short list of things to probe next -->

## Confirmed facts
<!-- durable, cross-level truths about this game -->
"""


class WorldModelAgent(Agent):

    def __init__(
        self,
        *,
        arc_env: Any,
        game_id: str,
        provider: str,
        model: str,
        effort: str | None,
        reasoning: str | None,
        workdir: Path,
        events: EventSink,
        resume: bool = False,
        executor: Any = None,
        claude_container: Any = None,
    ) -> None:
        super().__init__(arc_env=arc_env, game_id=game_id)
        self.executor = executor
        self.events = events
        self.workdir = Path(workdir).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._src_dir = _SRC_DIR
        self._seed_workdir()
        self.resume = bool(resume)
        self.provider = provider
        self._tool_specs = TOOL_SPECS
        self._system_prompt = SYSTEM_PROMPT

        if provider == "claude":
            self.driver = ClaudeDriver(model=model, effort=effort, cwd=str(self.workdir),
                                       container=claude_container)
            if not self.driver.available():
                raise RuntimeError(
                    "`claude` CLI not found on PATH. Install @anthropic-ai/claude-code and log in."
                )
        elif provider == "codex-cli":
            from .codex_cli_driver import CodexCliDriver

            self.driver = CodexCliDriver(model=model, reasoning=reasoning, cwd=str(self.workdir))
            if not self.driver.available():
                raise RuntimeError(
                    "`codex` CLI not found or no login (~/.codex-arc-agent*/auth.json or "
                    "~/.codex/auth.json); npm i -g @openai/codex and run codex login first."
                )
        else:
            raise ValueError(f"unknown provider: {provider}")

        self.timeline: list[TimeStep] = []
        self.code: Optional[str] = None
        self.world = None
        self.last_surprise = ""
        self.last_outcome = ""
        self.last_suggestion = ""
        self._turn = 0
        self._current_level = 0
        self.resumed_transitions = 0
        self._last_turn_no_commit = False
        self._consecutive_no_commit = 0
        self._consecutive_zero_output = 0
        self._just_cleared: Optional[int] = None
        self.entry_grids: dict[int, np.ndarray] = {}
        if self.resume:
            self._resume_state()
            self.action_counter = len(self.timeline)

    def is_done(self, latest: FrameData) -> bool:
        return latest.state is GameState.WIN

    def main(self) -> None:
        first = self.frames[-1]
        self.events.emit(RunStarted(
            game_id=self.game_id,
            provider=self.provider,
            model=getattr(self.driver, "model", None),
            max_actions=self.MAX_ACTIONS,
            win_levels=first.win_levels,
            workdir=str(self.workdir),
            resumed=self.resume,
            resumed_transitions=self.resumed_transitions,
        ))
        if self.resume and self.timeline:
            self._replay_into_env()
        while not self.is_done(self.frames[-1]) and self.action_counter < self.MAX_ACTIONS:
            latest = self._convert_raw_frame_data(self.arc_env.observation_space)
            self._current_level = latest.levels_completed
            self._note_entry(latest.levels_completed, self._grid(latest))

            if latest.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
                self._reset(latest)
                continue

            legal = self._legal(latest)
            if not legal:
                self._reset(latest)
                continue

            plan, reason = self._deliberate(latest, legal)
            if not plan:
                if self._consecutive_no_commit >= self._NO_COMMIT_LIMIT:
                    logger.warning("[harness] %d consecutive no-commit turns — circuit breaker: "
                                   "ending run (model wedged).", self._consecutive_no_commit)
                    break
                if self._consecutive_no_commit == self._SESSION_RESET_AT:
                    self.driver.reset_session()
                    logger.info("[harness] %d no-commit turns — dropped the resume session; next "
                                "turn restarts fresh with the full observation.",
                                self._consecutive_no_commit)
                continue
            self._execute(plan, latest, reason)
            self._snapshot_latest()

        self.cleanup()

    def _execute(self, plan: list[ActionStep], latest: FrameData, reason: str) -> None:
        self.last_surprise = ""
        cur = latest
        before_levels = cur.levels_completed
        start_level, start_state = cur.levels_completed, cur.state.name
        executed = 0
        stop = "ran the whole committed plan"
        for (action_id, x, y) in plan:
            if self.action_counter >= self.MAX_ACTIONS:
                stop = f"the action budget ({self.MAX_ACTIONS}) was exhausted — rest dropped"
                break
            if action_id != GameAction.RESET.value and not self._action_legal(
                self._legal(cur), action_id, x, y
            ):
                stop = f"the next planned action {action_id} was no longer legal — rest dropped"
                break
            before_grid = self._grid(cur)
            action = self._mk(action_id, x, y, reason)
            frame = self.take_action(action)
            if frame is None:
                stop = "the environment returned no frame — execution halted"
                break
            self.append_frame(frame)
            self.action_counter += 1
            executed += 1

            ts = TimeStep(action=action, x=x, y=y, frame=frame, before_levels=before_levels)
            self._record(ts)

            if self.is_done(frame):
                stop = "you WON the game"
                break
            if frame.state is GameState.GAME_OVER:
                stop = "you DIED (game over) — RESET to retry the level"
                break
            if frame.levels_completed != before_levels:
                stop = f"you cleared a level (advanced {before_levels}→{frame.levels_completed})"
                break
            if self.world is None or before_grid is None:
                stop = "no world model to self-check yet, so only this one step ran (exploring)"
                break
            if not self._model_step_ok(before_grid, ts):
                where = f"action {action_id}" + (f" @({x},{y})" if x is not None else "")
                note = (
                    f"world model MISPREDICTED the step just taken ({where}); the rest of the "
                    f"committed plan was dropped. "
                    f"Run backtest to see the mismatch and fix the model before planning again.")
                pred_grid = None
                out = self._model_predict_for(ts, before_grid)
                if out is not None:
                    pred_grid = encode_grid(out[0])
                self.last_surprise = note
                self.events.emit(ModelMispredicted(
                    turn=self._turn, step_index=len(self.timeline) - 1,
                    surprise=note, predicted=pred_grid, actual=encode_grid(ts.after)))
                stop = "the world model MISPREDICTED this step (see NOTE) — rest of the plan dropped"
                break
            before_levels, cur = frame.levels_completed, frame
        self._set_last_outcome(plan, reason, executed, stop, start_level, start_state)

    @staticmethod
    def _fmt_step(step: ActionStep) -> str:
        a, x, y = step
        if a == 0:
            return "RESET"
        if a == 6 and x is not None:
            return f"6@{x},{y}"
        return str(a)

    def _set_last_outcome(self, plan: list[ActionStep], reason: str, executed: int,
                          stop: str, start_level: int, start_state: str) -> None:
        last = self.frames[-1]
        intent = " ".join((reason or "").split())
        if len(intent) > 220:
            intent = intent[:217] + "…"
        plan_str = " ".join(self._fmt_step(s) for s in plan)
        animation = ""
        if executed:
            base = len(self.timeline) - executed
            counts = [
                (base + offset, len(ts.ticks))
                for offset, ts in enumerate(self.timeline[base:])
            ]
            if any(count for _, count in counts):
                animation = (
                    " Intermediate animation frames per step: "
                    + " ".join(f"#{index}:{count}" for index, count in counts)
                    + '. Use read_history detail="animation" with that index to inspect them.'
                )
        self.last_outcome = (
            f"committed {len(plan)} action(s) [{plan_str}] — executed {executed}; "
            f"stopped because {stop}. "
            f"Net: level {start_level}→{last.levels_completed}, "
            f"state {start_state}→{last.state.name}."
            + animation
            + (f' Your stated intent was: "{intent}"' if intent else "")
        )

    def _record(self, ts: TimeStep) -> None:
        self.timeline.append(ts)
        frame = ts.frame
        before = self.timeline[-2].after if len(self.timeline) >= 2 else None
        ticks, ticks_truncated = encode_ticks(before, frame.frame)
        self.events.emit(ActionTaken(
            turn=self._turn,
            step_index=len(self.timeline) - 1,
            action=ts.action_id,
            x=ts.x,
            y=ts.y,
            grid=encode_grid(ts.after),
            level_up=bool(ts.level_up),
            dead=bool(ts.dead),
            win=bool(ts.win),
            state=frame.state.name,
            level=frame.levels_completed,
            ticks=ticks,
            ticks_truncated=ticks_truncated,
        ))
        if ts.level_up:
            self._note_entry(ts.levels_after, ts.after)
            self._snapshot_model(ts.levels_after)
            self._just_cleared = ts.before_levels

    def _snapshot_model(self, reached_level: int) -> None:
        if not self.code:
            return
        cleared = max(0, int(reached_level) - 1)
        snap_dir = self.workdir / "snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)
        path = snap_dir / f"cleared_level_{cleared}.py"
        try:
            path.write_text(self.code, encoding="utf-8")
            logger.warning("snapshot: saved model that cleared level %d -> %s", cleared, path)
        except Exception as e:
            logger.warning("snapshot: failed to write %s: %s", path, e)

    def _snapshot_latest(self) -> None:
        try:
            self.driver.export_sessions(self.workdir / "sessions")
        except Exception as e:
            logger.warning("snapshot: failed to dump sessions: %s", e)
        self._git_commit(f"turn {self._turn} · L{self._current_level} · env_step {self.action_counter}")
        jc = self._just_cleared
        if jc is not None:
            self._git_tag(f"cleared_level_{jc}")
            self._just_cleared = None

    def _note_entry(self, level: int, grid) -> None:
        if grid is not None:
            self.entry_grids.setdefault(int(level), np.asarray(grid, dtype=np.int8))

    def _ground(self, level: Optional[int]) -> None:
        if self.world is not None:
            try:
                self.world.set_entry_grid(
                    self.entry_grids.get(int(level)) if level is not None else None,
                    level,
                )
            except Exception:
                pass

    def _rollout(self, level: int, end: int):
        return rollout_state(self.world, self.timeline, self.entry_grids, level, end)

    def _model_predict_for(self, ts: TimeStep, before_grid: np.ndarray):
        world = self.world
        if world is None or before_grid is None:
            return None
        idx = len(self.timeline) - 1
        if self.executor is not None:
            try:
                return world.predict_step(
                    self.timeline, ts.before_levels, self.entry_grids, idx,
                    before_grid, ts.action_id, ts.x, ts.y)
            except Exception:
                return None
        use_alarm = threading.current_thread() is threading.main_thread()

        class _Budget(Exception):
            pass

        if use_alarm:
            def _on_alarm(signum, frame):
                raise _Budget()
            prev = signal.signal(signal.SIGALRM, _on_alarm)
            signal.alarm(int(STEP_CHECK_S))
        try:
            state = self._rollout(ts.before_levels, idx)
            self._ground(ts.before_levels)
            pred, info, _ = world.predict(state, before_grid, ts.action_id, ts.x, ts.y)
        except _Budget:
            return None
        except Exception:
            return None
        finally:
            if use_alarm:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, prev)
        return pred, info

    def _model_step_ok(self, before_grid: np.ndarray, ts: TimeStep) -> bool:
        if self.world is None or before_grid is None:
            return False
        if ts.action_id == 0:
            return True
        out = self._model_predict_for(ts, before_grid)
        if out is None:
            return False
        pred, info = out
        for flag in ("level_up", "dead", "win"):
            if bool(info.get(flag)) != bool(getattr(ts, flag)):
                return False
        if not (ts.level_up or ts.dead or ts.win):
            after = ts.after
            if after is not None and (pred.shape != after.shape or not np.array_equal(pred, after)):
                return False
        return True

    _ZERO_OUTPUT_BACKOFF_BASE = 2.0
    _ZERO_OUTPUT_MAX_BACKOFF = 60.0

    @classmethod
    def _zero_output_backoff(cls, n: int) -> float:
        if n <= 1:
            return 0.0
        return min(cls._ZERO_OUTPUT_BACKOFF_BASE * (2 ** min(n - 2, 20)),
                   cls._ZERO_OUTPUT_MAX_BACKOFF)

    _SESSION_RESET_AT = 3
    _NO_COMMIT_LIMIT = 10
    _NO_COMMIT_WARN = (
        "[harness WARNING] Your previous turn ended WITHOUT calling commit_actions, so NO action "
        "was executed and the game state is UNCHANGED — what you see now is the SAME observation as "
        "last turn (the game did not advance). Every turn MUST end by calling commit_actions with at "
        "least one action. If you needed an extra think/model-only turn that is fine, but nothing in "
        "the game moves until you commit — decide and commit now."
    )
    _NO_COMMIT_SEVERE_WARN = (
        '[harness] you haven\'t commit action, please call "commit_actions" tool.'
    )

    def _deliberate(self, latest: FrameData, legal: list[int]) -> tuple[list[ActionStep], str]:
        self._turn += 1
        turn = self._turn
        self._current_level = latest.levels_completed
        self._note_entry(latest.levels_completed, self._grid(latest))
        grid = self._grid(latest)
        self.events.emit(TurnStarted(
            turn=turn,
            env_step=self.action_counter,
            state=latest.state.name,
            level=latest.levels_completed,
            win_levels=latest.win_levels,
            legal=list(legal),
            grid=encode_grid(grid),
            has_world_model=self.world is not None,
            surprise=self.last_surprise,
        ))
        box = ToolBox(self, grid, legal)
        committed: dict[str, Any] = {"plan": None, "reason": "", "suggestion": ""}
        call_n = {"i": 0}
        t_turn_start = time.time()

        def on_tool_call(name: str, args: dict) -> tuple[str, bool, bool]:
            if name.startswith("mcp__"):
                name = name.split("__")[-1]
            call_n["i"] += 1
            call_id = f"t{turn}.{call_n['i']}"
            self.events.emit(ToolStarted(
                turn=turn, call_id=call_id, name=name,
                args=dict(args) if isinstance(args, dict) else {"_": args},
            ))

            def done(output: str, is_error: bool, stop: bool) -> tuple[str, bool, bool]:
                self.events.emit(ToolFinished(
                    turn=turn, call_id=call_id, name=name, output=output, is_error=is_error,
                ))
                return (output, is_error, stop)

            if name == "commit_actions":
                if committed["plan"] is not None:
                    return done("Already committed this turn — end your turn now.", False, True)
                plan, err = self._parse_commit(args, legal)
                if plan is not None:
                    committed["plan"], committed["reason"] = plan, str(args.get("reason", ""))
                    committed["suggestion"] = str(args.get("suggestion", ""))
                    return done(
                        f"Committed {len(plan)} action(s). Stop now — end your turn, do not call more tools.",
                        False, True,
                    )
                return done(f"ERROR: {err}", True, False)
            out = box.dispatch(name, args)
            return done(out, out.startswith("ERROR"), False)

        spoke = {"yes": False}
        _emit_event = make_on_event(self.provider, self.events, turn)

        def on_event(msg: dict) -> None:
            if isinstance(msg, dict) and msg.get("type") == "stream_event":
                spoke["yes"] = True
            _emit_event(msg)

        def build_user_message(continuing: bool) -> dict:
            content = observation_content(self, latest, grid, legal, continuing=continuing)
            if self._last_turn_no_commit:
                nc_warn = (self._NO_COMMIT_SEVERE_WARN
                           if self._consecutive_no_commit >= 2 else self._NO_COMMIT_WARN)
                content = content + [{"type": "text", "text": nc_warn}]
            return {"role": "user", "content": content}

        try:
            self.driver.run_turn(self._system_prompt, build_user_message, self._tool_specs, on_tool_call, on_event)
        except Exception as e:
            logger.warning("driver turn failed: %s: %s", type(e).__name__, e)

        if committed["plan"] is not None:
            self.last_suggestion = committed["suggestion"]
            self._last_turn_no_commit = False
            self._consecutive_no_commit = 0
            self._consecutive_zero_output = 0
            self.events.emit(TurnCommitted(
                turn=turn,
                plan=[[a, x, y] for (a, x, y) in committed["plan"]],
                reason=committed["reason"],
            ))
            return committed["plan"], committed["reason"]
        self._last_turn_no_commit = True
        if spoke["yes"] or call_n["i"] > 0:
            self._consecutive_no_commit += 1
            self._consecutive_zero_output = 0
        else:
            self._consecutive_zero_output += 1
            back_off = self._zero_output_backoff(self._consecutive_zero_output)
            logger.warning(
                "[harness] turn %d produced no model output at all (took %.1fs) — treating as an "
                "infrastructure failure, NOT a wedged model; circuit-breaker count stays at %d. "
                "Consecutive zero-output turns: %d → backing off %.0fs before retrying.",
                turn, time.time() - t_turn_start, self._consecutive_no_commit,
                self._consecutive_zero_output, back_off,
            )
            if back_off:
                time.sleep(back_off)
        self.events.emit(TurnFallback(
            turn=turn,
            reason="ended without commit_actions — no action taken, game state unchanged (warned next turn)",
        ))
        return [], "no commit — no action taken"

    def _parse_commit(self, args: dict, legal: list[int]) -> tuple[Optional[list[ActionStep]], str]:
        acts = args.get("actions")
        if not isinstance(acts, list) or not acts:
            return None, "actions must be a non-empty list of {action, x?, y?}"
        out: list[ActionStep] = []
        for idx, item in enumerate(acts):
            if not isinstance(item, dict):
                return None, f"action item #{idx} must be an object, got {item!r}"
            try:
                a = int(item.get("action"))
            except (TypeError, ValueError):
                return None, f"action item #{idx}: 'action' must be an int"
            x = item.get("x")
            y = item.get("y")
            x = int(x) if x is not None else None
            y = int(y) if y is not None else None
            if idx == 0 and not self._action_legal(legal, a, x, y):
                return None, f"first action {a} not legal now; legal={legal} (action 6 needs x,y)"
            if a == 6 and (x is None or y is None):
                return None, f"action item #{idx}: action 6 (click) requires x and y"
            if a == 6 and not (0 <= x <= 63 and 0 <= y <= 63):
                return None, f"action item #{idx}: action 6 (click) x,y must be in 0..63, got ({x},{y})"
            out.append((a, x, y))
        return out, ""

    def _reset(self, latest: Optional[FrameData] = None) -> None:
        before_levels = latest.levels_completed if latest is not None else 0
        action = self._mk(GameAction.RESET.value, None, None, "reset")
        frame = self.take_action(action)
        if frame is not None:
            self.append_frame(frame)
            self._record(
                TimeStep(
                    action=action,
                    x=None,
                    y=None,
                    frame=frame,
                    before_levels=before_levels,
                )
            )
        self.action_counter += 1

    @staticmethod
    def _grid(frame: FrameData) -> Optional[np.ndarray]:
        return np.asarray(frame.frame[-1], dtype=np.int8) if frame.frame else None

    @staticmethod
    def _legal(frame: FrameData) -> list[int]:
        return [int(i) for i in frame.available_actions if int(i) != 0]

    @staticmethod
    def _action_legal(legal: list[int], action_id: int, x: Optional[int], y: Optional[int]) -> bool:
        action_id = int(action_id)
        if action_id == 0:
            return True
        if action_id == 6:
            return 6 in legal and x is not None and y is not None
        return action_id in legal

    @staticmethod
    def _mk(action_id: int, x: Optional[int], y: Optional[int], reason: str = "") -> GameAction:
        action = GameAction.from_id(int(action_id))
        if action.is_complex() and x is not None and y is not None:
            action.set_data({"x": int(x), "y": int(y)})
        action.reasoning = {"world_model": {"reason": reason}}
        return action

    def _seed_workdir(self) -> None:
        p = self.workdir / "notes.md"
        if not p.exists():
            p.write_text(_NOTES_SEED, encoding="utf-8")
        gi = self.workdir / ".gitignore"
        if not gi.exists():
            gi.write_text("session_live/\n__pycache__/\n*.pyc\n", encoding="utf-8")
        gd = self.workdir / ".git"
        if not gd.exists() or (gd.is_dir() and not any(gd.iterdir())):
            self._git("init", "-q")
            self._git_commit("seed (start of level 0)")
            self._git_tag("start")

    def _git(self, *args: str) -> None:
        try:
            subprocess.run(["git", "-C", str(self.workdir),
                            "-c", "user.name=world_model", "-c", "user.email=world_model@local",
                            *args],
                           check=False, capture_output=True)
        except Exception as e:
            logger.warning("git %s failed: %s", args, e)

    def _git_commit(self, msg: str) -> None:
        self._git("add", "-A")
        self._git("commit", "-q", "--allow-empty", "-m", msg)

    def _git_tag(self, name: str) -> None:
        self._git("tag", "-f", name)

    def _resume_state(self) -> None:
        code_path = self.workdir / "world_model.py"
        if code_path.is_file():
            try:
                code = code_path.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning("resume: cannot read %s: %s", code_path, e)
                code = ""
            if code.strip():
                self.code = code
                if self.executor is not None:
                    from sandbox.world_proxy import WorldModelProxy
                    ld = self.executor.world_load(code)
                    if ld.get("loaded"):
                        self.world = WorldModelProxy(
                            self.executor, stateful=ld["stateful"], has_is_goal=bool(ld.get("has_is_goal")),
                            has_win_condition=bool(ld.get("has_win_condition")))
                    else:
                        logger.warning("resume: world load failed in container (%s); world=None",
                                       ld.get("error"))
                        self.world = None
                else:
                    try:
                        self.world = CodeWorldModel(code)
                    except Exception as e:
                        logger.warning(
                            "resume: world model failed to load (%s: %s); kept as text, world=None",
                            type(e).__name__, e,
                        )
                        self.world = None
        self.timeline = self._replay_timeline(self.workdir / "events.jsonl")
        self.resumed_transitions = len(self.timeline)
        try:
            self.driver.import_sessions(self.workdir / "sessions")
        except Exception as e:
            logger.warning("resume: failed to import sessions: %s", e)
        logger.warning(
            "resume from %s: %d transition(s), world model %s, up to turn %d",
            self.workdir, self.resumed_transitions,
            "loaded" if self.world is not None else ("text-only" if self.code else "none"),
            self._turn,
        )

    def _replay_into_env(self) -> None:
        n = len(self.timeline)
        logger.warning("resume: replaying %d recorded action(s) into a fresh env to restore game state…", n)
        try:
            init = self._convert_raw_frame_data(self.arc_env.observation_space)
            if init is not None and init.state is GameState.NOT_FINISHED:
                self._note_entry(init.levels_completed, self._grid(init))
        except Exception:
            pass
        matched = 0
        for i, ts in enumerate(self.timeline):
            action = self._mk(ts.action_id, ts.x, ts.y, "resume-replay")
            frame = self.take_action(action)
            if frame is None:
                logger.warning(
                    "resume replay: env returned no frame at step %d (action %d); "
                    "stopping replay — live game restored only up to step %d.",
                    i, ts.action_id, i,
                )
                return
            self.append_frame(frame)
            self._note_entry(frame.levels_completed, self._grid(frame))
            rec, got = ts.after, self._grid(frame)
            grid_ok = (rec is None and got is None) or (
                rec is not None and got is not None
                and rec.shape == got.shape and np.array_equal(rec, got)
            )
            meta_ok = frame.levels_completed == ts.levels_after and frame.state is ts.state
            if grid_ok and meta_ok:
                matched += 1
                continue
            logger.warning(
                "resume replay DIVERGED at step %d (action %d): grid_match=%s, "
                "level got=%d want=%d, state got=%s want=%s. Stopping replay; the live "
                "game may not fully match recorded history (agent will re-orient from its model).",
                i, ts.action_id, grid_ok,
                frame.levels_completed, ts.levels_after, frame.state.name, ts.state.name,
            )
            return
        final = self.frames[-1]
        logger.warning(
            "resume: replay complete — %d/%d steps reproduced exactly; "
            "live game restored to level %d, state %s.",
            matched, n, final.levels_completed, final.state.name,
        )

    def _replay_timeline(self, path: Path) -> list[TimeStep]:
        if not path.is_file():
            return []
        timeline: list[TimeStep] = []
        max_turn = 0
        win_levels = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = ev.get("kind")
            if kind == "run_started":
                win_levels = int(ev.get("win_levels") or 0) or win_levels
                continue
            if kind != "action_taken":
                continue
            max_turn = max(max_turn, int(ev.get("turn") or 0))
            grid = ev.get("grid")
            try:
                state = GameState[str(ev.get("state") or "NOT_PLAYED")]
            except KeyError:
                state = GameState.NOT_PLAYED
            level = int(ev.get("level") or 0)
            before_levels = max(0, level - 1) if ev.get("level_up") else level
            frames: list = []
            if isinstance(grid, list):
                before_grid = timeline[-1].after if timeline else None
                try:
                    frames = [
                        tick.tolist()
                        for tick in decode_ticks(before_grid, ev.get("ticks"))
                    ]
                except Exception:
                    frames = []
                frames.append(grid)
            frame = FrameData(
                game_id=self.game_id,
                frame=frames,
                state=state,
                levels_completed=level,
                win_levels=win_levels,
            )
            action = GameAction.from_id(int(ev.get("action") or 0))
            timeline.append(
                TimeStep(
                    action=action,
                    x=ev.get("x"),
                    y=ev.get("y"),
                    frame=frame,
                    before_levels=before_levels,
                )
            )
            if ev.get("level_up") and isinstance(grid, list):
                self._note_entry(level, grid)
        self._turn = max(self._turn, max_turn)
        return timeline

    def save_code(self) -> None:
        if self.code is not None:
            (self.workdir / "world_model.py").write_text(self.code, encoding="utf-8")

    def cleanup(self) -> None:
        self.save_code()
        latest = self.frames[-1]
        self.events.emit(RunFinished(
            state=self.state.name,
            levels=self.levels_completed,
            win_levels=latest.win_levels,
            actions=self.action_counter,
            transitions=len(self.timeline),
            has_world_model=self.world is not None,
        ))
        self.events.close()
