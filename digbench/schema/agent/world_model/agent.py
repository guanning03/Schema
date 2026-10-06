from __future__ import annotations

import json
import logging
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Optional

from agent.agent import Agent
from env.dig_env import DigState, Transition

from .codex_driver import CodexDriver
from .events import (
    ActionTaken,
    EventSink,
    ModelMispredicted,
    RunControl,
    RunFinished,
    RunStarted,
    ToolFinished,
    ToolStarted,
    TurnCommitted,
    TurnFallback,
    TurnStarted,
)
from .stream_norm import make_on_event
from .timestep import TimeStep, segment_entry
from .tools import ToolBox, build_system_prompt, observation_content, tool_specs
from .world import score_step

logger = logging.getLogger(__name__)
_STEP_CHECK_S = 60.0
_SRC_DIR = Path(__file__).resolve().parent


def private_seed_path(workdir: Path) -> Path:
    wd = Path(workdir).resolve()
    return wd.parent / ".private" / f"{wd.name}.json"


def stash_private_seed(workdir: Path, game_id: Any, session_id: Any, seed: Any) -> Optional[Path]:
    """The bench session seed fixes the server's RNG stream: keep it out of the agent-readable run
    records (events.jsonl, run.json) and store it in a sidecar OUTSIDE the workdir."""
    if seed is None:
        return None
    p = private_seed_path(workdir)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"game_id": game_id, "session_id": session_id, "seed": seed,
                                 "workdir": str(Path(workdir).resolve())}), encoding="utf-8")
        return p
    except OSError as e:
        logger.warning("could not stash session seed at %s: %s", p, e)
        return None


_RANDOM_AGENT_SEED = '''\
"""Starter scaffold seeded into the world_model workdir at startup.

A trivial random-action policy. It is NOT executed by the framework — it is here
for you to READ and EXTEND (with read_file / write_file / edit_file) while you
build a real solution. Actions are STRING tokens; only the currently legal ones
may be submitted, and the legal set can change from state to state.
"""

from __future__ import annotations

import random


def choose_action(obs, legal_actions):
    """Pick an action for the current observation.

    obs:           the observation string (the rendered screen), or None.
    legal_actions: list[str] of currently-legal action tokens.
    returns:       one action token from legal_actions (None if nothing is legal).
    """
    if not legal_actions:
        return None
    return random.choice(legal_actions)
'''

_NOTES_SEED = """\
# Notes — your living scratchpad (shown to you every turn).
# Keep it CONCISE; edit and PRUNE stale entries with write_file / edit_file as you learn.

## Action semantics (confirmed / guessed)
<!-- e.g. "confirmed: action '2' moves the * right"; "guess: 'w' waits one tick" -->

## Current level
<!-- same vs previous levels; new symbols; goal hypothesis; current plan; step/life budget -->

## Hypotheses to test
<!-- short list of things to probe next (prefer creative mode for risky probes) -->

## Confirmed facts
<!-- durable, cross-level truths about this game -->
"""


class WorldModelAgent(Agent):
    def __init__(
        self,
        *,
        env: Any,
        game_id: str,
        model: str,
        reasoning: str | None,
        workdir: Path,
        events: EventSink,
        resume: bool = False,
        session_id: str | None = None,
        max_hours: float | None = None,
        model_name: str | None = None,
    ) -> None:
        super().__init__(env, game_id=game_id)
        self.events = events
        self.workdir = Path(workdir).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._src_dir = _SRC_DIR
        self._seed_workdir()
        self.resume = bool(resume)
        self.session_id: Optional[str] = session_id
        self.provider = "codex"
        self.model_name = model_name
        self.max_hours = max_hours
        self._tool_specs = tool_specs()
        self._system_prompt = build_system_prompt()

        self.driver = CodexDriver(model=model, reasoning=reasoning)
        if not self.driver.available():
            raise RuntimeError(
                "no codex login found (~/.codex-arc-agent*/auth.json or ~/.codex/auth.json); "
                "run `CODEX_HOME=~/.codex-arc-agent codex login` first."
            )

        self.timeline: list[TimeStep] = []
        self.code: Optional[str] = None
        self.world = None
        self.description: Optional[str] = None
        self.last_reason = ""
        self.last_surprise = ""
        self.last_outcome = ""
        self.last_suggestion = ""
        self.stop_reason = ""
        self._turn = 0
        self._segment = 0
        self.entries: dict[int, dict] = {}
        self.resumed_transitions = 0
        self._bench_failed = False
        self._last_turn_no_commit = False
        self._consecutive_no_commit = 0
        self._consecutive_no_output = 0
        self._consecutive_toggle_only = 0
        self._toggle_warned = False
        self._just_cleared: Optional[int] = None
        self._last_backtest: Optional[dict] = None
        self._commits_since_backtest = 0
        self._level_entries: dict[int, str] = {}
        self._restart_note = ""
        self._hist_only_steps = 0
        if self.resume:
            self._resume_state()
            self.action_counter = len(self.timeline)

    def is_done(self, latest: Transition) -> bool:
        return bool(latest.done or latest.state.done)

    def _time_cap_hit(self) -> bool:
        return bool(self.max_hours) and (time.time() - self.timer) > float(self.max_hours) * 3600.0

    def main(self) -> None:
        self.timer = time.time()
        if self.resume:
            server_t = self.open(session_id=self.session_id)
            self._reconcile_resume(server_t)
        else:
            self.open()
            self._segment = 0
            self.entries = {0: segment_entry(self.frames[-1])}
        self._note_level_entry(self.frames[-1])
        self.session_id = getattr(self.env, "session_id", self.session_id)
        self.description = getattr(self.env, "description", None)
        first = self.frames[-1]
        stash_private_seed(self.workdir, self.game_id, self.session_id, getattr(self.env, "seed", None))
        self.events.emit(RunStarted(
            game_id=self.game_id,
            provider=self.provider,
            model=getattr(self.driver, "model", None),
            max_actions=self.MAX_ACTIONS,
            max_level=first.state.max_level,
            workdir=str(self.workdir),
            resumed=self.resume,
            resumed_transitions=self.resumed_transitions,
            session_id=self.session_id,
            base_url=str(getattr(self.env, "base_url", "")),
            model_name=self.model_name,
            description=self.description,
            seed=None,
            framework_version=getattr(self.env, "framework_version", None),
            initial_state=first.state.to_dict(),
            initial_step_index=int(first.step_index),
            initial_levels_beaten=int(first.levels_beaten),
        ))
        while True:
            latest = self.frames[-1]
            if self.is_done(latest):
                self.stop_reason = "completed" if latest.state.won else "game_over"
                break
            if self.action_counter >= self.MAX_ACTIONS:
                self.stop_reason = "max_actions"
                break
            if self._time_cap_hit():
                self.stop_reason = "time_cap"
                break
            if self._bench_failed:
                self.stop_reason = "bench_failure"
                break

            legal = self._legal(latest)
            if not legal:
                try:
                    latest = self.env.get_session()
                    self.frames[-1] = latest
                except Exception as e:
                    logger.warning("get_session failed: %s", e)
                if self.is_done(latest):
                    continue
                if not self._legal(latest):
                    self.stop_reason = "no_legal_actions"
                    break

            plan, reason = self._deliberate(latest, legal)
            self.last_reason = reason
            if self._consecutive_toggle_only >= self._TOGGLE_STALL_LIMIT:
                logger.warning("[harness] %d consecutive toggle-only commits (accepted + rejected) — "
                               "parking; ending run.", self._consecutive_toggle_only)
                self.stop_reason = "toggle_stall"
                break
            if not plan:
                if self._consecutive_no_commit >= self._NO_COMMIT_LIMIT:
                    logger.warning("[harness] %d consecutive no-commit turns — circuit breaker: "
                                   "ending run (model wedged).", self._consecutive_no_commit)
                    self.stop_reason = "no_commit_wedged"
                    break
                if self._consecutive_no_output:
                    if self._consecutive_no_output >= self._NO_OUTPUT_LIMIT:
                        logger.warning("[harness] %d consecutive zero-output turns — the LLM driver is "
                                       "failing (auth / limits / CLI); ending run.", self._consecutive_no_output)
                        self.stop_reason = "driver_failure"
                        break
                    delay = min(2.0 ** self._consecutive_no_output, 120.0)
                    logger.warning("[harness] zero-output turn #%d — retrying in %.0fs",
                                   self._consecutive_no_output, delay)
                    time.sleep(delay)
                continue
            if self._toggle_only(plan, latest):
                self._consecutive_toggle_only += 1
            else:
                self._consecutive_toggle_only = 0
            self._execute(plan, latest, reason)
            self._snapshot_latest()

        self.cleanup()

    _OUTAGE_MAX_S = 1800.0

    def take_action(self, action: str, *, reasoning: Optional[str] = None) -> Optional[Transition]:
        env = self.env
        before_idx = getattr(env, "step_index", None)
        frame = super().take_action(action, reasoning=reasoning)
        if frame is not None or before_idx is None or not hasattr(env, "get_session"):
            return frame
        deadline = time.time() + self._OUTAGE_MAX_S
        delay = 15.0
        while time.time() < deadline:
            logger.warning("[harness] bench step %r failed — polling the session in %.0fs "
                           "(giving up at %s)", action, delay,
                           time.strftime("%H:%M:%S", time.localtime(deadline)))
            time.sleep(delay)
            delay = min(delay * 1.5, 120.0)
            try:
                t = env.get_session()
            except Exception as e:
                logger.warning("[harness] session poll failed: %s", e)
                continue
            srv_idx = int(t.step_index)
            if srv_idx == int(before_idx) + 1:
                logger.warning("[harness] the server had applied step %r (index %d) — adopting its state",
                               action, srv_idx)
                try:
                    return Transition(state=t.state, step_index=t.step_index, action=str(action),
                                      invalid_action=bool(t.invalid_action), events=list(t.events),
                                      levels_beaten=int(t.levels_beaten), done=bool(t.done))
                except Exception:
                    return t
            if srv_idx == int(before_idx):
                frame = super().take_action(action, reasoning=reasoning)
                if frame is not None:
                    return frame
                continue
            logger.warning("[harness] server step_index %d is neither %d nor %d — cannot recover",
                           srv_idx, before_idx, int(before_idx) + 1)
            return None
        return None

    def _execute(self, plan: list[str], latest: Transition, reason: str) -> None:
        self.last_surprise = ""
        self._restart_note = ""
        cur = latest
        start_level, start_status = cur.state.level, cur.state.status
        start_legal = list(self._legal(cur))
        bench_events: list[str] = []
        executed = 0
        stop = "ran the whole committed plan"
        for action in plan:
            action = str(action)
            if self.action_counter >= self.MAX_ACTIONS:
                stop = f"the action budget ({self.MAX_ACTIONS}) was exhausted — rest dropped"
                break
            if self._time_cap_hit():
                stop = "the wall-clock cap was reached — rest dropped"
                break
            if not self._action_legal(self._legal(cur), action):
                stop = (f"the next planned action {action!r} was no longer legal "
                        f"(legal: {self._legal(cur)!r}) — rest dropped")
                break
            frame = self.take_action(action, reasoning=reason or None)
            if frame is None:
                self._bench_failed = True
                stop = "the bench call FAILED (network/protocol error) — execution halted; the run will stop and can be --resume'd"
                break
            self.append_frame(frame)
            self.action_counter += 1
            executed += 1

            ts = TimeStep(action=action, before=cur, after=frame, segment=self._segment)
            self._record(ts)
            bench_events += self._describe_events(ts)

            if ts.invalid:
                stop = (f"the bench REJECTED action {action!r} as not legal (state unchanged) — "
                        "rest dropped")
                break
            model_note = ""
            if ts.terminal and self.world is not None and ts.scored:
                if not self._model_step_ok(ts):
                    model_note = " — NOTE: your world model did NOT predict this outcome (see NOTE)"
            if self.is_done(frame):
                stop = ("you WON the game" if frame.state.won else "GAME OVER — all lives lost") + model_note
                break
            if ts.level_up:
                stop = f"you cleared level {ts.level_before} (now on level {ts.level_after}){model_note}"
                break
            if ts.life_lost:
                why = "the level's step budget ran out" if ts.budget_exhausted else "a losing state was reached"
                stop = (f"you LOST A LIFE ({why}); level {ts.level_after} restarted"
                        + (f" — {self._restart_note}" if self._restart_note else "")
                        + (f"; {ts.lives_after} lives left" if ts.lives_after is not None else "")
                        + model_note)
                break
            if ts.mode_switch:
                if ts.challenge_outcome == "solved":
                    stop = (f"the creative challenge was SOLVED and you are back in {ts.mode_after} mode "
                            f"(its final board is scored by run_backtest){model_note}")
                elif ts.challenge_outcome == "failed":
                    stop = (f"the creative challenge FAILED (no life lost) and you are back in "
                            f"{ts.mode_after} mode (its final board is scored by run_backtest){model_note}")
                else:
                    stop = f"mode switched to {ts.mode_after} (new board)"
                break
            if ts.transition:
                stop = f"the bench reported a transition: {ts.transition}"
                break
            if self.world is None:
                stop = "no world model to self-check yet, so only this one step ran (exploring)"
                break
            if not self._model_step_ok(ts):
                stop = "the world model MISPREDICTED this step (see NOTE) — rest of the plan dropped"
                break
            cur = frame
        self._set_last_outcome(plan, reason, executed, stop, start_level, start_status,
                               start_legal=start_legal, bench_events=bench_events)

    @staticmethod
    def _describe_events(ts: TimeStep) -> list[str]:
        out: list[str] = []
        for ev in ts.events:
            t = str(ev.get("type", "") or "")
            if not t or t in ("creative_entered", "creative_exited"):
                continue
            if t == "action_unlocked" and ev.get("action") is not None:
                out.append(f"action_unlocked({ev.get('action')!r})")
            else:
                out.append(t)
        return out

    def _set_last_outcome(self, plan: list[str], reason: str, executed: int,
                          stop: str, start_level: int, start_status: str, *,
                          start_legal: Optional[list[str]] = None,
                          bench_events: Optional[list[str]] = None) -> None:
        last = self.frames[-1]
        intent = " ".join((reason or "").split())
        if len(intent) > 220:
            intent = intent[:217] + "…"
        st = last.state
        counters = (f"level {start_level}→{st.level}, status {start_status}→{st.status}"
                    + (f", lives {st.lives_left}" if st.lives_left is not None else "")
                    + (f", steps_remaining {st.steps_remaining}" if st.steps_remaining is not None else "")
                    + (f", mode {st.mode}" if st.mode else ""))
        extras: list[str] = []
        if start_legal is not None:
            now_legal = self._legal(last)
            added = [a for a in now_legal if a not in start_legal]
            gone = [a for a in start_legal if a not in now_legal]
            if added or gone:
                extras.append("Legal actions CHANGED during this commit: "
                              + (f"+{added!r} " if added else "") + (f"-{gone!r}" if gone else ""))
        if bench_events:
            extras.append("Bench events: " + ", ".join(bench_events))
        if self._hist_only_steps:
            extras.append(f"{self._hist_only_steps} step(s) matched on the current frame but not on the "
                          "embedded history prefix (accepted, not a misprediction)")
            self._hist_only_steps = 0
        self.last_outcome = (
            f"committed {len(plan)} action(s) {plan!r} — executed {executed}; "
            f"stopped because {stop}. Net: {counters}."
            + (" " + ". ".join(extras) + "." if extras else "")
            + (f' Your stated intent was: "{intent}"' if intent else "")
        )

    def _record(self, ts: TimeStep) -> None:
        self.timeline.append(ts)
        st = ts.state
        self.events.emit(ActionTaken(
            turn=self._turn,
            step_index=len(self.timeline) - 1,
            server_step_index=ts.server_step_index,
            action=ts.action,
            observation=ts.after,
            legal=list(st.actions),
            level=st.level,
            max_level=st.max_level,
            lives_left=st.lives_left,
            steps_remaining=st.steps_remaining,
            max_steps=st.max_steps,
            mode=st.mode,
            status=st.status,
            done=bool(ts.frame.done),
            transition=st.transition,
            invalid=ts.invalid,
            levels_beaten=int(ts.frame.levels_beaten),
            events=ts.events,
            level_up=ts.level_up,
            life_lost=ts.life_lost,
            dead=ts.dead,
            win=ts.win,
            mode_switch=ts.mode_switch,
            budget_exhausted=ts.budget_exhausted,
            segment=ts.segment,
            state=st.to_dict(),
        ))
        if ts.terminal and not ts.invalid and not ts.frame.done:
            self._segment += 1
            self.entries[self._segment] = segment_entry(ts.frame)
        if ts.level_up:
            self._snapshot_model(ts.level_before)
            self._just_cleared = ts.level_before
            self._note_level_entry(ts.frame)
        elif ts.life_lost and not ts.frame.done:
            self._restart_note = self._restart_identity(ts)
            self._note_level_entry(ts.frame)

    def _note_level_entry(self, frame: Transition) -> None:
        st = frame.state
        if (st.mode or "survival") == "creative":
            return
        self._level_entries.setdefault(int(st.level), str(st.observation))

    def _restart_identity(self, ts: TimeStep) -> str:
        prev = self._level_entries.get(int(ts.level_after))
        if prev is None or (ts.state.mode or "survival") == "creative":
            return ""
        if ts.after == prev:
            return "the restart board is IDENTICAL to this level's first entry board"
        return "the restart board DIFFERS from this level's first entry board"

    def _snapshot_model(self, cleared_level: int) -> None:
        if not self.code:
            return
        snap_dir = self.workdir / "snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)
        path = snap_dir / f"cleared_level_{int(cleared_level)}.py"
        try:
            path.write_text(self.code, encoding="utf-8")
            logger.warning("snapshot: saved model that cleared level %d -> %s", cleared_level, path)
        except Exception as e:
            logger.warning("snapshot: failed to write %s: %s", path, e)

    def _snapshot_latest(self) -> None:
        try:
            self._dump_sessions()
        except Exception as e:
            logger.warning("snapshot: latest-turn session dump failed: %s", e)
        lv = self.frames[-1].state.level if self.frames else "?"
        self._git_commit(f"turn {self._turn} · L{lv} · env_step {self.action_counter}")
        jc = self._just_cleared
        if jc is not None:
            self._git_tag(f"cleared_level_{jc}")
            self._just_cleared = None

    def _dump_sessions(self) -> None:
        export = getattr(self.driver, "export_sessions", None)
        if not callable(export):
            return
        try:
            export(self.workdir / "sessions")
        except Exception as e:
            logger.warning("snapshot: failed to dump sessions: %s", e)

    def _rollout(self, segment: int, end: int):
        from .world import rollout_state
        return rollout_state(self.world, self.timeline, self.entries, segment, end)

    def _model_predict_for(self, ts: TimeStep):
        world = self.world
        if world is None or ts.before_obs is None:
            return None
        try:
            idx = self.timeline.index(ts)
        except ValueError:
            idx = len(self.timeline)
        budget = _STEP_CHECK_S
        use_alarm = threading.current_thread() is threading.main_thread()

        class _Budget(Exception):
            pass

        if use_alarm:
            def _on_alarm(signum, frame):
                raise _Budget()
            prev = signal.signal(signal.SIGALRM, _on_alarm)
            signal.alarm(int(budget))
        try:
            state = self._rollout(ts.segment, idx)
            world.set_entry(self.entries.get(ts.segment))
            pred, info, nst = world.predict(state, ts.before_obs, ts.action)
        except _Budget:
            return None
        except Exception:
            return None
        finally:
            if use_alarm:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, prev)
        return pred, info, nst

    def _model_step_ok(self, ts: TimeStep) -> bool:
        if self.world is None:
            return False
        if not ts.scored:
            return True
        out = self._model_predict_for(ts)
        if out is None:
            self._emit_mispredict(ts, None, None, "the world model raised/timed out (cannot verify)")
            return False
        pred, info, nst = out
        sc = score_step(self.world, pred, info or {}, nst, ts)
        if sc["errors"]:
            self._emit_mispredict(ts, pred, info, "; ".join(sc["errors"]))
            return False
        if "obs_hist" in sc["kinds"]:
            self._hist_only_steps += 1
        return True

    def _emit_mispredict(self, ts: TimeStep, pred: Optional[str], info: Optional[dict], why: str) -> None:
        self.last_surprise = (
            f"world model MISPREDICTED the step just taken (action {ts.action!r}): {why}. "
            "The rest of the committed plan was dropped. Run run_backtest to see the mismatch "
            "and fix the model before planning again."
        )
        self.events.emit(ModelMispredicted(
            turn=self._turn,
            step_index=len(self.timeline) - 1,
            surprise=self.last_surprise,
            predicted=pred,
            actual=ts.after,
            predicted_flags={k: bool((info or {}).get(k)) for k in ("level_up", "life_lost", "dead", "win")},
            actual_flags=ts.flags(),
        ))

    _last_backtest: Optional[dict] = None
    _commits_since_backtest = 0
    _restart_note = ""
    _hist_only_steps = 0

    @property
    def _level_entries(self) -> dict:
        d = self.__dict__.get("_level_entries_d")
        if d is None:
            d = self.__dict__["_level_entries_d"] = {}
        return d

    @_level_entries.setter
    def _level_entries(self, value: dict) -> None:
        self.__dict__["_level_entries_d"] = dict(value)

    _NO_COMMIT_LIMIT = 10
    _NO_OUTPUT_LIMIT = 30
    _TOGGLE_STALL_WARN_AT = 5
    _TOGGLE_STALL_LIMIT = 15
    _TOGGLE_STALL_WARN = (
        "[harness WARNING] Your last {n} commits were ONLY the creative-mode toggle. Toggling back and "
        "forth without playing (\"parking\") makes no progress: the game only advances through "
        "actions in survival mode. Further toggle-only commits are REJECTED, and the run is ENDED after "
        "{limit} consecutive toggle-only commits. Either experiment in creative mode (take actions "
        "there), or act in survival mode — if the level looks lost, the honest outcome is to play it out."
    )
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

    def _deliberate(self, latest: Transition, legal: list[str]) -> tuple[list[str], str]:
        self._turn += 1
        turn = self._turn
        st = latest.state
        obs = st.observation
        self.events.emit(TurnStarted(
            turn=turn,
            env_step=self.action_counter,
            status=st.status,
            level=st.level,
            max_level=st.max_level,
            lives_left=st.lives_left,
            steps_remaining=st.steps_remaining,
            max_steps=st.max_steps,
            mode=st.mode,
            transition=st.transition,
            legal=list(legal),
            observation=obs,
            has_world_model=self.world is not None,
            surprise=self.last_surprise,
        ))
        box = ToolBox(self, obs, legal)
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
                plan, err = self._parse_commit(args if isinstance(args, dict) else {}, legal, latest)
                if plan is not None:
                    committed["plan"], committed["reason"] = plan, str(args.get("reason", ""))
                    committed["suggestion"] = str(args.get("suggestion", ""))
                    return done(
                        f"Committed {len(plan)} action(s). Stop now — end your turn, do not call more tools.",
                        False, True,
                    )
                return done(f"ERROR: {err}", True, False)
            out = box.dispatch(name, args if isinstance(args, dict) else {})
            return done(out, out.startswith("ERROR"), False)

        on_event = make_on_event(self.events, turn)

        def build_user_message(continuing: bool) -> dict:
            content = observation_content(self, latest, legal, continuing=continuing)
            if self._last_turn_no_commit:
                nc_warn = (self._NO_COMMIT_SEVERE_WARN
                           if self._consecutive_no_commit >= 2 else self._NO_COMMIT_WARN)
                content = content + [{"type": "text", "text": nc_warn}]
            if self._consecutive_toggle_only >= self._TOGGLE_STALL_WARN_AT:
                content = content + [{"type": "text", "text": self._TOGGLE_STALL_WARN.format(
                    n=self._consecutive_toggle_only, limit=self._TOGGLE_STALL_LIMIT)}]
            return {"role": "user", "content": content}

        turn_result = None
        try:
            turn_result = self.driver.run_turn(
                self._system_prompt, build_user_message, self._tool_specs, on_tool_call, on_event)
        except Exception as e:
            logger.warning("driver turn failed: %s: %s", type(e).__name__, e)

        usage = getattr(turn_result, "usage_total", None) or getattr(turn_result, "usage", None) or {}
        usage = dict(usage) if isinstance(usage, dict) else {}
        if committed["plan"] is not None:
            self.last_suggestion = committed["suggestion"]
            self._last_turn_no_commit = False
            self._consecutive_no_commit = 0
            self._commits_since_backtest += 1
            self.events.emit(TurnCommitted(
                turn=turn,
                plan=list(committed["plan"]),
                reason=committed["reason"],
                usage=usage,
            ))
            return committed["plan"], committed["reason"]
        self._last_turn_no_commit = True
        if call_n["i"] > 0:
            self._consecutive_no_commit += 1
            self._consecutive_no_output = 0
        else:
            self._consecutive_no_output += 1
            kind = getattr(turn_result, "error_kind", None)
            err = (getattr(turn_result, "stderr", "") or getattr(turn_result, "final_text", "") or "").strip()
            logger.warning(
                "[harness] turn %d produced no model output at all (took %.1fs; driver error_kind=%s) — "
                "infrastructure failure #%d, not a wedged model. %s",
                turn, time.time() - t_turn_start, kind, self._consecutive_no_output,
                ("driver says: " + err[-400:].replace("\n", " | ")) if err else "",
            )
        self.events.emit(TurnFallback(
            turn=turn,
            reason="ended without commit_actions — no action taken, game state unchanged (warned next turn)",
            usage=usage,
        ))
        return [], "no commit — no action taken"

    def _parse_commit(self, args: dict, legal: list[str],
                      latest: Optional[Transition] = None) -> tuple[Optional[list[str]], str]:
        acts = args.get("actions")
        if isinstance(acts, str):
            acts = [acts]
        if not isinstance(acts, list) or not acts:
            return None, "actions must be a non-empty list of {action: <token>} (or plain token strings)"
        out: list[str] = []
        for idx, item in enumerate(acts):
            if isinstance(item, dict):
                a = item.get("action")
            else:
                a = item
            if a is None or isinstance(a, (list, dict)):
                return None, f"action item #{idx} must be an action token string, got {item!r}"
            a = str(a)
            if idx == 0 and not self._action_legal(legal, a):
                return None, f"first action {a!r} is not legal now; legal={legal!r}"
            out.append(a)
        if latest is None:
            return out, ""
        err = self._gate_commit(out, latest, accept_risk=bool(args.get("accept_risk")))
        if err:
            return None, err
        return out, ""

    _RISK_SIM_BUDGET_S = 20.0

    def _gate_commit(self, plan: list[str], latest: Transition, *, accept_risk: bool) -> str:
        st = latest.state
        toggle = st.creative_toggle
        in_survival = (st.mode or "survival") != "creative"
        lives = st.lives_left
        last_life = in_survival and lives is not None and lives <= 1

        if self._toggle_only(plan, latest):
            if self._consecutive_toggle_only >= self._TOGGLE_STALL_WARN_AT:
                self._consecutive_toggle_only += 1
                return (f"REJECTED: this would be toggle-only commit #{self._consecutive_toggle_only} in a "
                        "row. Toggling back and forth makes no progress — commit at least one real "
                        "action: experiment in creative mode, or play the survival board (if the level "
                        "looks lost, play it out: the budget running out costs one life and restarts it).")
            return ""

        n_surv = len(plan)
        if toggle is not None and toggle in plan:
            n_surv = plan.index(toggle)
        if in_survival and st.steps_remaining is not None and n_surv > int(st.steps_remaining):
            if not accept_risk:
                return (f"REJECTED: the plan spends {n_surv} survival step(s) but only "
                        f"{st.steps_remaining} remain on this level — it would run the budget out and "
                        "lose a life before finishing. Shorten it (leave a safety margin), probe in "
                        "creative mode instead, or pass accept_risk=true if running the budget out is "
                        "your deliberate choice.")
            if last_life:
                return (f"REJECTED: the plan spends {n_surv} survival step(s) with only "
                        f"{st.steps_remaining} left and this is your LAST life — running the budget out "
                        "now ends the game. Commit a plan that fits the budget.")

        risk = self._predicted_loss(plan, latest)
        if risk is not None:
            k, tok, note = risk
            if last_life:
                return (f"REJECTED: your own world model predicts this plan LOSES A LIFE at action #{k} "
                        f"({tok!r}){note} and this is your LAST life — a predicted loss now is game over. "
                        "If you believe the model is wrong here, fix it and run_backtest first; otherwise "
                        "commit a plan the model predicts safe.")
            if not accept_risk:
                return (f"REJECTED: your own world model predicts this plan LOSES A LIFE at action #{k} "
                        f"({tok!r}){note}; lives_left={lives}. A predicted loss is only worth it as a "
                        "deliberate, informative experiment with lives to spare (and only when creative "
                        "mode cannot answer the same question). If that is the case, re-commit with "
                        "accept_risk=true and state in `reason` exactly what the outcome will teach you.")

        if last_life:
            if self.world is None:
                if len(plan) > 1 and not accept_risk:
                    return ("REJECTED: LAST life and no world model installed — commit ONE action at a "
                            "time (or accept_risk=true).")
            elif not self._model_verified() and not accept_risk:
                bt = self._last_backtest
                if bt is None or bt.get("code") != self._code_hash():
                    why = "no run_backtest has been run on the installed model"
                else:
                    why = f"the last run_backtest on it had {bt.get('mismatches')} mismatch(es)"
                return ("REJECTED: LAST life — commit only plans your model predicts safe AFTER a green "
                        f"full-scope run_backtest on the installed model ({why}). Run run_backtest now "
                        "(fix the model if it is red), then commit; accept_risk=true overrides this "
                        "only if you have a specific reason.")
        return ""

    def _predicted_loss(self, plan: list[str], latest: Transition) -> Optional[tuple]:
        world = self.world
        if world is None or not hasattr(world, "predict"):
            return None
        st = latest.state
        toggle = st.creative_toggle
        try:
            seg = self._segment
            world.set_entry(self.entries.get(seg))
            state = self._rollout(seg, len(self.timeline))
        except Exception:
            return None
        obs = st.observation
        deadline = time.time() + self._RISK_SIM_BUDGET_S
        for k, a in enumerate(plan, start=1):
            if toggle is not None and a == toggle:
                break
            if time.time() > deadline:
                break
            try:
                pred, info, state = world.predict(state, obs, a)
            except Exception:
                break
            info = info or {}
            if info.get("life_lost") or info.get("dead"):
                return k, a, (" (dead: game over)" if info.get("dead") else "")
            if info.get("level_up") or info.get("win"):
                break
            obs = pred
        return None

    def _code_hash(self) -> Optional[str]:
        if not self.code:
            return None
        import hashlib
        return hashlib.sha1(self.code.encode("utf-8")).hexdigest()

    def _model_verified(self) -> bool:
        bt = self._last_backtest
        return bool(bt and bt.get("code") == self._code_hash() and bt.get("mismatches") == 0
                    and bt.get("full_scope"))

    def note_backtest(self, *, ok: int, total: int, mismatches: int, win_verified: int,
                      full_scope: bool, breakdown: Optional[dict] = None) -> None:
        self._last_backtest = {"code": self._code_hash(), "ok": ok, "total": total,
                               "mismatches": mismatches, "win_verified": win_verified,
                               "full_scope": bool(full_scope),
                               "breakdown": {k: list(v) for k, v in (breakdown or {}).items()}}
        self._commits_since_backtest = 0

    def harness_notes(self, latest: Transition, legal: list[str]) -> list[str]:
        st = latest.state
        notes: list[str] = []
        if self.world is None:
            notes.append("Model check: no world model installed yet.")
        else:
            bt = self._last_backtest
            level_now = int(st.level)
            if bt is None or bt.get("code") != self._code_hash():
                status = "NOT backtested since its last edit"
            else:
                bd = bt.get("breakdown") or {}
                by = " · ".join(f"{mode} L{lv} {o}/{t}" for (mode, lv), (o, t)
                                in sorted(bd.items(), key=lambda kv: (kv[0][1], kv[0][0])))
                if bt.get("mismatches") == 0:
                    status = f"GREEN {bt['ok']}/{bt['total']}{'' if bt.get('full_scope') else ' (scoped)'}"
                else:
                    status = f"RED {bt['mismatches']} mismatch(es) of {bt['total']}"
                if by:
                    status += f" [{by}]"
                if st.mode is not None and bt.get("full_scope") and bd.get(("survival", level_now), [0, 0])[1] == 0:
                    status += f" — no survival-mode evidence on level {level_now} yet"
            wv = (bt or {}).get("win_verified")
            win_note = ""
            if getattr(self.world, "has_win_condition", False):
                if bt is None or bt.get("code") != self._code_hash():
                    win_note = "; win rule: unverified (backtest it)"
                elif not wv:
                    win_note = "; win rule: NEVER confirmed on a real completion (a hypothesis)"
                else:
                    win_note = f"; win rule confirmed on {wv} real completion(s)"
            else:
                win_note = "; no is_win_condition defined"
            notes.append(f"Model check: installed model is {status}{win_note}; "
                         f"{self._commits_since_backtest} commit(s) since the last backtest.")
        level = int(st.level)
        counts: dict[str, int] = {}
        creative_steps = 0
        for t in self.timeline:
            if t.invalid or t.is_toggle:
                continue
            if t.level_before == level:
                counts[t.action] = counts.get(t.action, 0) + 1
                if (t.mode_before or "survival") == "creative":
                    creative_steps += 1
        ever = {t.action for t in self.timeline if not t.invalid}
        tog = st.creative_toggle
        cov = ", ".join(f"{a!r}:{counts.get(a, 0)}" for a in legal if a != tog)
        never = [a for a in legal if a != tog and a not in ever]
        line = f"Action coverage on level {level} (all modes): {cov or '(none yet)'}"
        if never:
            line += f"; NEVER tried anywhere in this game: {never!r}"
        notes.append(line + ".")
        if st.mode is not None and (tog is not None or st.in_creative):
            notes.append(f"Creative steps used on level {level}: {creative_steps}.")
        return notes

    @staticmethod
    def _toggle_only(plan: list[str], latest: Transition) -> bool:
        tog = latest.state.creative_toggle
        return bool(plan) and tog is not None and all(str(a) == tog for a in plan)

    @staticmethod
    def _legal(frame: Transition) -> list[str]:
        return [str(a) for a in frame.state.actions]

    @staticmethod
    def _action_legal(legal: list[str], action: str) -> bool:
        return str(action) in legal

    def _seed_workdir(self) -> None:
        for name, content in (("random_agent.py", _RANDOM_AGENT_SEED), ("notes.md", _NOTES_SEED)):
            p = self.workdir / name
            if not p.exists():
                p.write_text(content, encoding="utf-8")
        gi = self.workdir / ".gitignore"
        if not gi.exists():
            gi.write_text(
                "session_live/\n__pycache__/\n*.pyc\n",
                encoding="utf-8",
            )
        gd = self.workdir / ".git"
        if not gd.exists() or (gd.is_dir() and not any(gd.iterdir())):
            self._git("init", "-q")
            self._git_commit("seed (start of the run)")
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
        from .world import CodeWorldModel

        code_path = self.workdir / "world_model.py"
        if code_path.is_file():
            try:
                code = code_path.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning("resume: cannot read %s: %s", code_path, e)
                code = ""
            if code.strip():
                self.code = code
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
        imp = getattr(self.driver, "import_sessions", None)
        if callable(imp):
            try:
                imp(self.workdir / "sessions")
            except Exception as e:
                logger.warning("resume: failed to import sessions: %s", e)
        logger.warning(
            "resume from %s: %d transition(s), world model %s, up to turn %d, session %s",
            self.workdir, self.resumed_transitions,
            "loaded" if self.world is not None else ("text-only" if self.code else "none"),
            self._turn, self.session_id,
        )

    def _reconcile_resume(self, server_t: Transition) -> None:
        if not self.timeline:
            if not self.entries:
                self._segment = 0
                self.entries = {0: segment_entry(server_t)}
            return
        last = self.timeline[-1]
        rec_idx = last.server_step_index
        srv_idx = int(server_t.step_index)
        same_obs = server_t.state.observation == last.after
        if srv_idx == rec_idx and same_obs:
            logger.warning("resume: server session matches the recorded history (step_index %d).", srv_idx)
            return
        detail = (f"recorded step_index {rec_idx} / obs {last.after!r} vs server step_index {srv_idx} / "
                  f"obs {server_t.state.observation!r}")
        if srv_idx > rec_idx:
            logger.warning("resume: the server is AHEAD of the recorded history (%d unrecorded step(s)) — "
                           "continuing from the server state; %s", srv_idx - rec_idx, detail)
        else:
            logger.warning("resume: server/recorded state MISMATCH — continuing from the server state; %s", detail)
        self.events.emit(RunControl(state="resume_mismatch", detail=detail))
        self._segment += 1
        self.entries[self._segment] = segment_entry(server_t)

    def _replay_timeline(self, path: Path) -> list[TimeStep]:
        if not path.is_file():
            return []
        initial: Optional[dict] = None
        rows: list[dict] = []
        max_turn = 0
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
                if initial is None:
                    initial = ev
                if ev.get("session_id") and not self.session_id:
                    self.session_id = str(ev["session_id"])
                if ev.get("description"):
                    self.description = str(ev["description"])
                continue
            if kind in ("turn_started", "turn_fallback", "turn_committed"):
                max_turn = max(max_turn, int(ev.get("turn") or 0))
            if kind == "action_taken":
                rows.append(ev)
                max_turn = max(max_turn, int(ev.get("turn") or 0))
        if initial is not None and initial.get("initial_state") is not None:
            prev = Transition(
                state=DigState.from_dict(initial["initial_state"]),
                step_index=int(initial.get("initial_step_index") or 0),
                levels_beaten=int(initial.get("initial_levels_beaten") or 0),
            )
        elif rows:
            prev = Transition(state=DigState.from_dict({"observation": "", "actions": []}), step_index=0)
        else:
            self._turn = max(self._turn, max_turn)
            return []
        seg = 0
        entries = {0: segment_entry(prev)}
        timeline: list[TimeStep] = []
        for ev in rows:
            st_d = ev.get("state")
            if not isinstance(st_d, dict):
                st_d = {
                    "observation": ev.get("observation"), "actions": ev.get("legal") or [],
                    "level": ev.get("level"), "max_level": ev.get("max_level"),
                    "lives_left": ev.get("lives_left"), "steps_remaining": ev.get("steps_remaining"),
                    "max_steps": ev.get("max_steps"), "mode": ev.get("mode"),
                    "status": ev.get("status"), "done": ev.get("done"), "transition": ev.get("transition"),
                }
            t = Transition(
                state=DigState.from_dict(st_d),
                step_index=int(ev.get("server_step_index") or 0),
                action=str(ev.get("action")),
                invalid_action=bool(ev.get("invalid")),
                events=list(ev.get("events") or []),
                levels_beaten=int(ev.get("levels_beaten") or 0),
                done=bool(ev.get("done")),
            )
            rec_seg = ev.get("segment")
            ts = TimeStep(action=str(ev.get("action")), before=prev, after=t,
                          segment=int(rec_seg) if rec_seg is not None else seg)
            timeline.append(ts)
            if ts.terminal and not ts.invalid and not t.done:
                seg = ts.segment + 1
                entries[seg] = segment_entry(t)
            prev = t
        self._turn = max(self._turn, max_turn)
        self._segment = seg
        self.entries = entries
        self.frames = [prev]
        self._level_entries = {}
        first_t = Transition(state=DigState.from_dict(initial["initial_state"]),
                             step_index=0) if initial is not None and initial.get("initial_state") else None
        if first_t is not None:
            self._note_level_entry(first_t)
        for ts in timeline:
            if ts.level_up or (ts.life_lost and not ts.frame.done):
                self._note_level_entry(ts.frame)
        return timeline

    def save_code(self) -> None:
        if self.code is not None:
            (self.workdir / "world_model.py").write_text(self.code, encoding="utf-8")

    def cleanup(self) -> None:
        self.save_code()
        latest = self.frames[-1] if self.frames else None
        st = latest.state if latest is not None else DigState()
        self.events.emit(RunFinished(
            status=st.status,
            level=st.level,
            max_level=st.max_level,
            levels_beaten=int(latest.levels_beaten) if latest is not None else 0,
            actions=self.action_counter,
            transitions=len(self.timeline),
            has_world_model=self.world is not None,
            stop_reason=self.stop_reason,
        ))
        self.events.close()
