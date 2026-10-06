from __future__ import annotations

from env.maze_env import MazeState, Transition

FLAG_NAMES = ("gem", "room_changed", "dead", "won")

_RESET_ACTIONS = ("reset",)


def _is_goto(action: str) -> bool:
    return " ".join(str(action or "").strip().lower().split()).startswith("go to level")


def meta_of(state: MazeState) -> dict:
    return {
        "room": state.room,
        "view": state.view,
        "yaw": int(state.yaw),
        "gems": int(state.gems),
        "visited_rooms": list(state.visited_rooms),
    }


class TimeStep:

    __slots__ = ("action", "before_t", "frame", "segment", "env_step_index")

    def __init__(self, *, action: str, before: Transition, after: Transition, segment: int) -> None:
        self.action: str = str(action)
        self.before_t: Transition = before
        self.frame: Transition = after
        self.segment: int = int(segment)
        self.env_step_index: int = int(after.step_index)

    @property
    def before(self) -> MazeState:
        return self.before_t.state

    @property
    def state(self) -> MazeState:
        return self.frame.state

    @property
    def before_obs(self) -> str:
        return self.before_t.state.observation

    @property
    def after(self) -> str:
        return self.frame.state.observation

    @property
    def before_meta(self) -> dict:
        return meta_of(self.before)

    @property
    def after_meta(self) -> dict:
        return meta_of(self.state)

    @property
    def gem(self) -> bool:
        return self.frame.gem_collected > 0

    @property
    def room_changed(self) -> bool:
        return bool(self.frame.room_changed)

    @property
    def dead(self) -> bool:
        return bool(self.state.dead)

    @property
    def won(self) -> bool:
        return bool(self.state.won)

    @property
    def invalid(self) -> bool:
        return bool(self.frame.invalid_action)

    @property
    def terminal(self) -> bool:
        if self.invalid:
            return False
        return (self.room_changed or self.dead or self.won
                or self.action in _RESET_ACTIONS or _is_goto(self.action))

    @property
    def flags(self) -> dict:
        return {"gem": self.gem, "room_changed": self.room_changed,
                "dead": self.dead, "won": self.won}


def segment_entries(timeline: "list") -> dict:
    entries: dict = {}
    for ts in timeline:
        seg = int(getattr(ts, "segment", 0))
        if seg not in entries:
            entries[seg] = {"obs": ts.before_obs, "meta": dict(getattr(ts, "before_meta", {}) or {})}
        nxt = seg + 1
        if getattr(ts, "terminal", False) and nxt not in entries:
            entries[nxt] = {"obs": ts.after, "meta": dict(getattr(ts, "after_meta", {}) or {})}
    return entries
