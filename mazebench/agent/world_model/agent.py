from __future__ import annotations

import copy
import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

from env.maze_env import MazeEnv, MazeState, Transition, canonical_action

from .events import (ActionTaken, EventSink, ModelMispredicted, RunFinished, RunStarted,
                     ScorecardUpdated, ToolFinished, ToolStarted, TurnCommitted, TurnFallback,
                     TurnStarted, TurnUsage)
from .timestep import TimeStep, meta_of, segment_entries
from .tools import (CURRENT_STATE_FILE, MODEL_FILE, NOTES_FILE, ROOM_NOTES_DIR, ToolBox, build_system_prompt,
                    clear_stale_plans, ensure_room_notes, expand_actions, history_record,
                    history_start_record, render_observation, tool_specs, write_current_board,
                    write_history)
from .world import CodeWorldModel, SegmentStarts, chain_segments, frames_match, rollout_state

logger = logging.getLogger(__name__)

_NOTES_SEED = """\
# World model & notes

(Empty. Write down every mechanic you confirm, as precisely as you can — this file and
world_model.py are the only things that survive context compaction.)

## Confirmed rules

## Current room

## Hypotheses to test
"""


class ResumeError(RuntimeError):
    pass


class WorldModelAgent:

    _NO_COMMIT_LIMIT = 10
    _ZERO_OUTPUT_BACKOFF_BASE = 2.0
    _ZERO_OUTPUT_MAX_BACKOFF = 60.0

    def __init__(self, *, env: MazeEnv, provider: str, model: Optional[str],
                 reasoning: Optional[str], effort: Optional[str], workdir: Path,
                 events: EventSink, max_hours: Optional[float], max_actions: int,
                 resume: bool = False, service_tier: Optional[str] = None) -> None:
        self.env = env
        self.events = events
        self.workdir = Path(workdir).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.provider = provider
        self.model = model
        self.max_hours = max_hours
        self.MAX_ACTIONS = int(max_actions)
        self.resume = bool(resume)

        self._tool_specs = tool_specs()
        self._system_prompt = build_system_prompt(self.MAX_ACTIONS)

        self.timeline: list = []
        self.frames: list = []
        self.world: Optional[CodeWorldModel] = None
        self.segment = 0
        self.turn = 0
        self.action_counter = 0
        self.last_outcome = ""
        self.last_suggestion = ""
        self._consecutive_no_commit = 0
        self._consecutive_zero_output = 0
        self._t0 = time.time()
        self._gem_snapshots = 0
        self.usage_total = {"input_tokens": 0, "cache_creation_input_tokens": 0,
                            "cache_read_input_tokens": 0, "output_tokens": 0}
        self.resumed_transitions = 0
        self._code_hash = ""
        self._rollout_cache: "dict | None" = None
        self._bfs_checkpoint: "dict | None" = None
        self.seg_starts = SegmentStarts()        # chained segment starts (carrying models only)

        self._seed_workdir()
        self.driver = self._make_driver(provider, model, reasoning, effort, service_tier)

    def _make_driver(self, provider: str, model, reasoning, effort, service_tier=None):
        if provider == "claude":
            from .claude_driver import ClaudeDriver
            d = ClaudeDriver(model=model, effort=effort, cwd=str(self.workdir))
        elif provider == "codex-cli":
            from .codex_cli_driver import CodexCliDriver
            d = CodexCliDriver(model=model, reasoning=reasoning, cwd=str(self.workdir),
                               service_tier=service_tier)
        else:
            raise ValueError(f"unknown provider {provider!r}")
        if not d.available():
            raise RuntimeError(f"provider {provider!r} is not available (CLI missing or not logged in)")
        return d

    def _seed_workdir(self) -> None:
        notes = self.workdir / NOTES_FILE
        if not notes.exists():
            notes.write_text(_NOTES_SEED, encoding="utf-8")
        (self.workdir / "snapshots").mkdir(exist_ok=True)
        (self.workdir / ROOM_NOTES_DIR).mkdir(exist_ok=True)

    _EXEC_ENV_KEEP = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TERM")

    def exec_env(self) -> dict:
        env = {k: os.environ[k] for k in self._EXEC_ENV_KEEP if k in os.environ}
        env.setdefault("PATH", "/usr/bin:/bin")
        env.update({"HOME": str(self.workdir), "PYTHONPATH": str(self.workdir), "TMPDIR": "/tmp"})
        return env

    def install_world_model(self, code: str) -> str:
        world = CodeWorldModel(code, workdir=self.workdir)
        self.world = world
        self._code_hash = world.version()
        self._rollout_cache = None
        if self.frames:
            try:
                st = self.frames[-1].state
                write_current_board(self.workdir, st.observation, meta_of(st), self._code_hash)
            except OSError:
                pass
        d = world.describe()
        bits = ["stateful (predict)" if d["stateful"] else "stateless (step)"]
        if d["has_bfs_goal"]:
            bits.append("is_bfs_goal defined")
        if d["local_modules"]:
            bits.append("imports " + ", ".join(d["local_modules"]))
        return "; ".join(bits)

    def segment_entries(self) -> dict:
        entries = segment_entries(self.timeline)
        if self.segment not in entries and self.frames:
            entries[self.segment] = {"obs": self.frames[-1].state.observation,
                                     "meta": meta_of(self.frames[-1].state)}
        return entries

    def rollout_now(self):
        if self.world is None:
            return None, meta_of(self.frames[-1].state) if self.frames else {}, [
                f"no world model installed — write {MODEL_FILE} first"]
        entries = self.segment_entries()
        end = len(self.timeline)
        cache = self._rollout_cache
        start, state, meta = 0, None, None
        if (cache is not None and cache["code"] == self._code_hash
                and cache["segment"] == self.segment and 0 < cache["n"] <= end):
            start = cache["n"]
            state = copy.deepcopy(cache["state"])
            meta = dict(cache["meta"])
        start_state = None
        if start <= 0 and self.world.carries:
            chain_segments(self.world, self.timeline, entries, self.seg_starts, self.segment)
            start_state = self.seg_starts.get(self.segment)
            err = self.seg_starts.errors.get(self.segment)
            if err:
                return None, dict((entries.get(self.segment) or {}).get("meta") or {}), [
                    f"{err} (start of the current segment)"]
        state, meta, errors = rollout_state(self.world, self.timeline, entries, self.segment, end,
                                            start=start, state=state, meta=meta,
                                            start_state=start_state)
        if errors:
            self._rollout_cache = None
        else:
            self._rollout_cache = {"code": self._code_hash, "segment": self.segment, "n": end,
                                   "state": copy.deepcopy(state), "meta": dict(meta)}
        return state, meta, errors

    def resume_from(self, events_path: Path) -> "Transition | None":
        rows = []
        if events_path.is_file():
            for line in events_path.read_text(errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        all_acts = [r for r in rows if r.get("kind") == "action_taken"]
        acts = [r for r in all_acts if not r.get("invalid")]
        if not all_acts:
            return None

        latest = self.env.replay([r["action"] for r in acts])

        started = next((r for r in rows if r.get("kind") == "run_started"), {})
        first = MazeState.from_dict(started.get("initial_state") or {})
        prev = Transition(state=first, step_index=0)
        for i, r in enumerate(all_acts):
            st = MazeState.from_dict(r.get("state") or {})
            esi = r.get("env_step_index")
            after = Transition(state=st, step_index=int(esi) if esi is not None else i + 1,
                               action=r.get("action"), invalid_action=bool(r.get("invalid")),
                               gem_collected=1 if r.get("gem") else 0,
                               room_changed=bool(r.get("room_changed")), raw={})
            ts = TimeStep(action=str(r.get("action") or ""), before=prev, after=after,
                          segment=int(r.get("segment") or 0))
            self.timeline.append(ts)
            self.frames.append(after)
            prev = after
        recs = [history_start_record(first)] + [history_record(i, int(r.get("turn") or 0), ts)
                                                for i, (r, ts) in enumerate(zip(all_acts, self.timeline))]
        self._write_history(recs, append=False)
        self.segment = int(all_acts[-1].get("segment") or 0)
        if self.timeline[-1].terminal:
            self.segment += 1
        self.action_counter = len(all_acts)
        self._gem_snapshots = int(all_acts[-1].get("gems") or 0)

        self.turn = max((int(r.get("turn") or 0) for r in rows), default=0)
        for u in rows:
            if u.get("kind") != "turn_usage":
                continue
            for k in self.usage_total:
                self.usage_total[k] += int(u.get(k) or 0)
        last_commit = next((r for r in reversed(rows) if r.get("kind") == "turn_committed"), None)
        if last_commit:
            self.last_suggestion = str(last_commit.get("suggestion") or "")
        last_row = next(r for r in reversed(rows) if r.get("kind") == "action_taken")
        rec = last_row.get("state") or {}
        if rec.get("observation") and rec["observation"] != latest.state.observation:
            logger.warning("resume: the replayed board differs from the recorded one")
        latest.state.transition = last_row.get("transition") or rec.get("transition")
        self.last_outcome = self._recorded_outcome(rows, latest) or (
            f"resumed here: {len(self.timeline)} transitions already executed "
            f"(gems {latest.state.gems}, {len(latest.state.visited_rooms)} rooms visited).")

        model = self.workdir / MODEL_FILE
        if model.is_file():
            try:
                self.install_world_model(model.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("resume: world model did not compile: %s", e)

        if not self._load_sessions():
            raise ResumeError(
                f"resume: could not restore the model's conversation from {self.workdir/'sessions'} "
                f"(provider {self.provider!r}). A run without it is not a continuation of the same "
                f"run.")
        self.resumed_transitions = len(self.timeline)
        return latest

    def _recorded_outcome(self, rows: list, latest: "Transition") -> str:
        commit = next((r for r in reversed(rows) if r.get("kind") == "turn_committed"), None)
        if commit is None:
            return ""
        turn, plan = commit.get("turn"), list(commit.get("plan") or [])
        done = [r for r in rows if r.get("kind") == "action_taken" and r.get("turn") == turn]
        if not plan or not done:
            return ""
        last = done[-1]
        dropped = ""
        if last.get("invalid"):
            dropped = (f"action {last.get('action')!r} was refused by the game; "
                       "the rest of the queue was dropped")
        else:
            mis = next((r for r in rows if r.get("kind") == "model_mispredicted"
                        and r.get("turn") == turn
                        and r.get("step_index") == last.get("step_index")), None)
            if mis is not None:
                dropped = (f"your world model MISPREDICTED the result of {last.get('action')!r} "
                           f"({mis.get('surprise')}); the rest of the queue was dropped")
            elif last.get("dead"):
                dropped = "the player died; the rest of the queue was dropped"
        return self._outcome(plan, len(done), dropped, latest)

    def main(self) -> None:
        latest = None
        if self.resume:
            latest = self.resume_from(self.workdir / "events.jsonl")
            if latest is not None:
                print(f"resumed from {self.workdir}: {len(self.timeline)} transition(s), "
                      f"turn {self.turn}, gems {latest.state.gems}, room {latest.state.room}, "
                      f"world model {'loaded' if self.world is not None else 'NONE'}")
        if latest is None:
            latest = self.env.start()
            self.frames.append(latest)
            self._write_history([history_start_record(latest.state)], append=False)
        self.events.emit(RunStarted(
            game_id="mazebench", provider=self.provider, model=self.model,
            max_actions=self.MAX_ACTIONS, workdir=str(self.workdir),
            resumed=self.resume, resumed_transitions=self.resumed_transitions,
            engine_root=str(self.env.engine), engine_commit=_git_head(self.env.engine),
            start_room=self.env.room, view=self.env.view, yaw=self.env.yaw,
            hide_names=self.env.hide_names, hide_names_seed=self.env.hide_names_seed,
            initial_state=latest.state.to_dict()))

        stop = "done"
        while True:
            if latest.state.done:
                stop = "won" if latest.state.won else "lost"
                break
            if self.action_counter >= self.MAX_ACTIONS:
                stop = "action_budget"
                if self.resumed_transitions and self.action_counter <= self.resumed_transitions:
                    print(f"resume: already at the action budget ({self.action_counter}"
                          f"/{self.MAX_ACTIONS}) — raise --steps to continue.")
                break
            if self.max_hours and (time.time() - self._t0) > self.max_hours * 3600:
                stop = "time_cap"
                break
            if self._consecutive_no_commit >= self._NO_COMMIT_LIMIT:
                stop = "no_commit_circuit_breaker"
                break

            self._emit_scorecard()
            plan, reason = self._deliberate(latest)
            if not plan:
                continue
            latest = self._execute(plan, latest, reason)

        card = self._emit_scorecard()
        st = latest.state
        self.events.emit(RunFinished(
            status=("won" if st.won else "lost" if st.lost else "stopped"),
            gems=st.gems, rooms_visited=len(st.visited_rooms), actions=self.action_counter,
            transitions=len(self.timeline), has_world_model=self.world is not None,
            stop_reason=stop, scorecard=card or None))

    def _emit_scorecard(self) -> dict:
        try:
            card = self.env.scorecard()
        except Exception:
            return {}
        if card:
            self.events.emit(ScorecardUpdated(scorecard=card))
        return card

    def _deliberate(self, latest: Transition) -> "tuple[list, str]":
        self.turn += 1
        legal = list(latest.legal_actions)
        st = latest.state
        try:
            ensure_room_notes(self.workdir, st.room)
        except OSError as e:
            logger.info("could not create the room notes file: %s", e)
        try:
            write_current_board(self.workdir, st.observation, meta_of(st),
                                self.world.version() if self.world is not None else None)
        except OSError as e:
            logger.info("could not write current_board.py: %s", e)
        self._write_current_state()
        self.events.emit(TurnStarted(
            turn=self.turn, env_step=latest.step_index, room=st.room, view=st.view, yaw=st.yaw,
            gems=st.gems, rooms_visited=len(st.visited_rooms), transition=st.transition,
            legal=legal, observation=st.observation, has_world_model=self.world is not None))
        box = ToolBox(self, st.observation, legal)
        committed: dict = {"plan": None, "reason": "", "suggestion": ""}
        calls = {"n": 0}
        t_turn = time.time()

        def on_tool_call(name: str, args: dict):
            calls["n"] += 1
            self.events.emit(ToolStarted(turn=self.turn, name=name, args=_clip_args(args)))
            if name == "commit_actions":
                plan, err = self._parse_commit(args)
                if err:
                    self.events.emit(ToolFinished(turn=self.turn, name=name, output=err, is_error=True))
                    return err, False, False
                committed["plan"] = plan
                committed["reason"] = str(args.get("reason") or "")
                committed["suggestion"] = str(args.get("suggestion") or "")
                msg = (f"Committed {len(plan)} action(s). Stop now — end your turn, "
                       "do not call more tools.")
                self.events.emit(ToolFinished(turn=self.turn, name=name, output=msg))
                return msg, True, True
            text = box.dispatch(name, args)
            self.events.emit(ToolFinished(turn=self.turn, name=name, output=_clip(text)))
            return text, False, False

        def build_user_message(_first: bool) -> dict:
            content = render_observation(self, latest, legal)
            if self.last_suggestion:
                content += f"\n\nYour note to yourself last turn: {self.last_suggestion}"
            if self._consecutive_no_commit:
                content += ("\n\n[harness WARNING] Your previous turn ended WITHOUT calling "
                            "commit_actions, so nothing was executed and the game did not advance. "
                            "Every turn must end with commit_actions.")
            return {"role": "user", "content": [{"type": "text", "text": content}]}

        result = None
        try:
            result = self.driver.run_turn(self._system_prompt, build_user_message,
                                          self._tool_specs, on_tool_call)
        except Exception as e:
            logger.warning("driver turn failed: %s: %s", type(e).__name__, e)
        self._record_usage(result)

        if committed["plan"] is not None:
            self.last_suggestion = committed["suggestion"]
            self._consecutive_no_commit = 0
            self._consecutive_zero_output = 0
            self.events.emit(TurnCommitted(turn=self.turn, plan=list(committed["plan"]),
                                           reason=committed["reason"],
                                           suggestion=committed["suggestion"]))
            self._dump_sessions()
            return committed["plan"], committed["reason"]

        if calls["n"] > 0:
            self._consecutive_no_commit += 1
            self._consecutive_zero_output = 0
        else:
            self._consecutive_zero_output += 1
            back = self._zero_output_backoff(self._consecutive_zero_output)
            logger.warning("[harness] turn %d produced no model output at all (%.1fs); "
                           "infrastructure failure, backing off %.0fs", self.turn,
                           time.time() - t_turn, back)
            if back:
                time.sleep(back)
        self.events.emit(TurnFallback(turn=self.turn,
                                      reason="ended without commit_actions — nothing executed"))
        return [], "no commit"

    def _write_current_state(self) -> None:
        """current_state.pkl = the model's state for the board in current_board.py (if picklable)."""
        import pickle
        p = self.workdir / CURRENT_STATE_FILE
        blob = None
        if self.world is not None and self.world.stateful:
            try:
                state, _meta, errs = self.rollout_now()
                blob = None if errs else pickle.dumps(state, 4)
            except Exception as e:
                logger.info("current_state.pkl not written: %s: %s", type(e).__name__, e)
        try:
            if blob is None:
                p.unlink(missing_ok=True)
                return
            tmp = p.with_name(p.name + ".tmp")
            tmp.write_bytes(blob)
            os.replace(tmp, p)
        except OSError as e:
            logger.info("could not write %s: %s", CURRENT_STATE_FILE, e)

    def segment_start(self, seg: int):
        """Carried start of `seg` if already chained for the installed model (never computes)."""
        return self.seg_starts.cached(self.world, seg)

    def _dump_sessions(self) -> None:
        try:
            self.driver.export_sessions(self.workdir / "sessions")
        except Exception as e:
            logger.info("session export failed: %s: %s", type(e).__name__, e)

    def _load_sessions(self) -> bool:
        if not (self.workdir / "sessions").is_dir():
            return False
        try:
            self.driver.import_sessions(self.workdir / "sessions")
            return True
        except Exception as e:
            logger.warning("session import failed: %s: %s", type(e).__name__, e)
            return False

    @staticmethod
    def _normalize_usage(usage: dict) -> dict:
        def _n(key: str) -> int:
            try:
                return int(usage.get(key) or 0)
            except (TypeError, ValueError):
                return 0

        if "cached_input_tokens" in usage or "cache_write_input_tokens" in usage:
            cached = _n("cached_input_tokens")
            uncached = max(0, _n("input_tokens") - cached)
            return {"input_tokens": uncached,
                    "cache_creation_input_tokens": _n("cache_write_input_tokens"),
                    "cache_read_input_tokens": cached,
                    "output_tokens": _n("output_tokens")}
        return {"input_tokens": _n("input_tokens"),
                "cache_creation_input_tokens": _n("cache_creation_input_tokens"),
                "cache_read_input_tokens": _n("cache_read_input_tokens"),
                "output_tokens": _n("output_tokens")}

    def _record_usage(self, result) -> None:
        usage = dict(getattr(result, "usage", None) or {}) if result is not None else {}
        if not usage:
            return
        norm = self._normalize_usage(usage)
        ev = TurnUsage(turn=self.turn, input_tokens=norm["input_tokens"],
                       cache_creation_input_tokens=norm["cache_creation_input_tokens"],
                       cache_read_input_tokens=norm["cache_read_input_tokens"],
                       output_tokens=norm["output_tokens"], raw=usage)
        self.usage_total["input_tokens"] += ev.input_tokens
        self.usage_total["cache_creation_input_tokens"] += ev.cache_creation_input_tokens
        self.usage_total["cache_read_input_tokens"] += ev.cache_read_input_tokens
        self.usage_total["output_tokens"] += ev.output_tokens
        self.events.emit(ev)

    @classmethod
    def _zero_output_backoff(cls, n: int) -> float:
        if n <= 1:
            return 0.0
        return min(cls._ZERO_OUTPUT_BACKOFF_BASE * (2 ** min(n - 2, 20)),
                   cls._ZERO_OUTPUT_MAX_BACKOFF)

    def _parse_commit(self, args: dict) -> "tuple[Optional[list], str]":
        acts, err = expand_actions(self.workdir, args.get("actions"))
        if err:
            return None, err
        out = []
        for i, a in enumerate(acts):
            try:
                out.append(canonical_action(str(a)))
            except ValueError as e:
                return None, f"ERROR: action #{i} — {e}"
        return out, ""

    def _execute(self, plan: list, latest: Transition, reason: str) -> Transition:
        executed = 0
        changed = 0
        dropped = ""
        state, meta, errs = (None, meta_of(latest.state), [])
        if self.world is not None:
            state, meta, errs = self.rollout_now()
            if errs:
                state = None

        for action in plan:
            pred = None
            pred_info: dict = {}
            if self.world is not None and not errs:
                try:
                    pred, pred_info, next_state = self.world.predict(
                        state, latest.state.observation, action, meta)
                except Exception as e:
                    pred, next_state = None, state
                    logger.info("model predict raised during execution: %s", e)
            else:
                next_state = state

            after = self.env.step(action)
            self.action_counter += 1
            ts = TimeStep(action=action, before=latest, after=after, segment=self.segment)
            self._record(ts)
            latest = after
            executed += 1
            if not after.invalid_action:
                changed += 1

            if after.invalid_action:
                dropped = f"action {action!r} was refused by the game; the rest of the queue was dropped"
                break

            surprise = ""
            if pred is not None:
                why = frames_match(pred, after.state.observation)
                flag_bad = [n for n in ("gem", "room_changed", "dead", "won")
                            if bool(pred_info.get(n)) != bool(ts.flags[n])]
                if why is not None or flag_bad:
                    surprise = why or ("flags differ: " + ", ".join(flag_bad))
                    self.events.emit(ModelMispredicted(
                        turn=self.turn, step_index=len(self.timeline) - 1, surprise=surprise,
                        predicted=pred, actual=after.state.observation,
                        predicted_flags={n: bool(pred_info.get(n))
                                         for n in ("gem", "room_changed", "dead", "won")},
                        actual_flags=dict(ts.flags)))

            if ts.terminal:
                self.segment += 1
                prev_state = next_state
                state, meta = None, meta_of(after.state)
                if self.world is not None:
                    entry = {"obs": after.state.observation, "meta": meta}
                    try:
                        self.world.set_entry(entry)
                        state = (self.world.init_state(entry["obs"], meta, prev_state)
                                 if self.world.carries else self.world.init_state(entry["obs"], meta))
                    except Exception:
                        state = None
            else:
                state = next_state
                meta = meta_of(after.state)

            if ts.gem:
                self._snapshot_model(after.state.gems)

            if surprise:
                dropped = (f"your world model MISPREDICTED the result of {action!r} "
                           f"({surprise}); the rest of the queue was dropped")
                break
            if ts.dead:
                dropped = "the player died; the rest of the queue was dropped"
                break
            if after.state.done:
                break

        self.last_outcome = self._outcome(plan, executed, dropped, latest)
        if changed:
            try:
                clear_stale_plans(self.workdir)
            except OSError:
                pass
        return latest

    def _outcome(self, plan: list, executed: int, dropped: str, latest: Transition) -> str:
        head = f"executed {executed}/{len(plan)} committed action(s)"
        if dropped:
            head += f" — {dropped}"
        st = latest.state
        return head + f". Now: room {st.room}, view {st.view}, yaw {st.yaw}, gems {st.gems}."

    def _record(self, ts: TimeStep) -> None:
        self.timeline.append(ts)
        self._write_history([history_record(len(self.timeline) - 1, self.turn, ts)], append=True)
        st = ts.state
        self.events.emit(ActionTaken(
            turn=self.turn, step_index=len(self.timeline) - 1, env_step_index=ts.env_step_index,
            action=ts.action, observation=st.observation, legal=list(st.actions),
            room=st.room, view=st.view, yaw=st.yaw, gems=st.gems,
            rooms_visited=len(st.visited_rooms), done=st.done, transition=st.transition,
            invalid=ts.invalid, gem=ts.gem, room_changed=ts.room_changed, dead=ts.dead,
            won=ts.won, segment=ts.segment, state=st.to_dict()))
        self.frames.append(ts.frame)

    def _write_history(self, records: list, *, append: bool) -> None:
        try:
            write_history(self.workdir, records, append=append)
        except OSError as e:
            logger.warning("history.jsonl write failed: %s", e)

    def _snapshot_model(self, gems: int) -> None:
        if gems <= self._gem_snapshots:
            return
        self._gem_snapshots = gems
        snaps = self.workdir / "snapshots"
        snaps.mkdir(exist_ok=True)
        for name, dst in ((MODEL_FILE, f"gem_{gems:03d}.py"), (NOTES_FILE, f"gem_{gems:03d}.md")):
            src = self.workdir / name
            if src.is_file():
                (snaps / dst).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    def cleanup(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass


def _git_head(root: Path) -> str:
    try:
        r = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        return (r.stdout or "").strip()
    except Exception:
        return ""


def _clip(text: str, cap: int = 4000) -> str:
    t = str(text or "")
    return t if len(t) <= cap else t[:cap] + f"\n… [clipped {len(t) - cap} chars]"


def _clip_args(args: dict) -> dict:
    out = {}
    for k, v in (args or {}).items():
        out[k] = _clip(v, 800) if isinstance(v, str) else v
    return out
