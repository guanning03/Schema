from __future__ import annotations

import re
from typing import Optional

from env.dig_env import DigState, Transition

FLAG_NAMES = ("level_up", "life_lost", "dead", "win")
_LEVEL_FAILED_RE = re.compile(r"\bLevel \d+ failed\b|\blost a life\b|\blife lost\b", re.I)
_FRAME_EVENTS = {"level_beaten": "level_beaten", "level_cleared": "level_beaten",
                 "level_failed": "level_failed",
                 "creative_solved": "creative_solved", "creative_failed": "creative_failed"}
_FINAL_BOARD_HEAD_RE = re.compile(r"^Final (?:board of level \d+|creative challenge board):\s*$")


class TimeStep:

    __slots__ = ("action", "before_t", "frame", "segment", "server_step_index")

    def __init__(self, *, action: str, before: Transition, after: Transition, segment: int) -> None:
        self.action: str = str(action)
        self.before_t: Transition = before
        self.frame: Transition = after
        self.segment: int = int(segment)
        self.server_step_index: int = int(after.step_index)

    @property
    def before(self) -> DigState:
        return self.before_t.state

    @property
    def state(self) -> DigState:
        return self.frame.state

    @property
    def before_obs(self) -> str:
        return self.before_t.state.observation

    @property
    def after(self) -> str:
        return self.frame.state.observation

    @property
    def status(self) -> str:
        return self.frame.state.status

    @property
    def events(self) -> list[dict]:
        return list(self.frame.events)

    @property
    def transition(self) -> Optional[str]:
        return self.frame.state.transition

    @property
    def level_before(self) -> int:
        return int(self.before.level)

    @property
    def level_after(self) -> int:
        return int(self.state.level)

    @property
    def mode_before(self) -> Optional[str]:
        return self.before.mode

    @property
    def mode_after(self) -> Optional[str]:
        return self.state.mode

    @property
    def lives_before(self) -> Optional[int]:
        return self.before.lives_left

    @property
    def lives_after(self) -> Optional[int]:
        return self.state.lives_left

    @property
    def steps_before(self) -> Optional[int]:
        return self.before.steps_remaining

    @property
    def steps_after(self) -> Optional[int]:
        return self.state.steps_remaining

    @property
    def invalid(self) -> bool:
        return bool(self.frame.invalid_action)

    @property
    def win(self) -> bool:
        return not self.invalid and self.state.status == "completed"

    @property
    def dead(self) -> bool:
        return not self.invalid and self.state.status == "game_over"

    @property
    def level_up(self) -> bool:
        if self.invalid:
            return False
        return int(self.frame.levels_beaten) > int(self.before_t.levels_beaten)

    @property
    def life_lost(self) -> bool:
        if self.invalid:
            return False
        if self.dead:
            return True
        lb, la = self.lives_before, self.lives_after
        if lb is not None and la is not None:
            return la < lb
        for ev in self.frame.events:
            if str(ev.get("type", "")).lower() in ("life_lost", "level_failed"):
                return True
        tr = self.state.transition or ""
        return bool(_LEVEL_FAILED_RE.search(tr))

    @property
    def mode_switch(self) -> bool:
        if self.invalid:
            return False
        mb, ma = self.mode_before, self.mode_after
        if mb is None or ma is None:
            return False
        return mb != ma

    @property
    def terminal(self) -> bool:
        return self.level_up or self.life_lost or self.dead or self.win or self.mode_switch

    @property
    def is_toggle(self) -> bool:
        tog = self.before.creative_toggle
        return tog is not None and self.action == str(tog)

    @property
    def challenge_outcome(self) -> Optional[str]:
        for ev in self.frame.events:
            t = str(ev.get("type", "")).lower()
            if t == "creative_solved":
                return "solved"
            if t == "creative_failed":
                return "failed"
        return None

    @property
    def final_frame(self) -> Optional[str]:
        if self.invalid:
            return None
        for ev in self.frame.events:
            if str(ev.get("type", "")).lower() in _FRAME_EVENTS and ev.get("frame") is not None:
                return str(ev["frame"])
        tr = self.state.transition
        if tr:
            lines = str(tr).split("\n")
            if len(lines) >= 3 and _FINAL_BOARD_HEAD_RE.match(lines[0].strip()):
                return "\n".join(lines[1:-1])
        return None

    @property
    def scored(self) -> bool:
        if self.invalid or self.is_toggle:
            return False
        if self.mode_switch and self.challenge_outcome is None:
            return False
        return True

    @property
    def budget_exhausted(self) -> bool:
        if not self.life_lost:
            return False
        for ev in self.frame.events:
            if str(ev.get("reason", "")).lower() in ("budget_exhausted", "steps_exhausted", "out_of_steps"):
                return True
        sb = self.steps_before
        if sb is not None and sb <= 1 and (self.mode_before or "survival") != "creative":
            return True
        return False

    def flags(self) -> dict:
        return {k: bool(getattr(self, k)) for k in FLAG_NAMES}

    def flag_names(self) -> list[str]:
        out = [k for k in FLAG_NAMES if getattr(self, k)]
        if self.mode_switch:
            out.append("mode_switch")
        if self.invalid:
            out.append("invalid")
        return out

    def __repr__(self) -> str:
        return (f"TimeStep(action={self.action!r} seg={self.segment} {self.before_obs!r} -> {self.after!r} "
                f"flags={self.flag_names() or ['none']})")


def segment_entry(t: Transition) -> dict:
    return {"obs": t.state.observation, "level": int(t.state.level), "mode": t.state.mode}
