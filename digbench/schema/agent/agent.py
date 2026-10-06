from __future__ import annotations

import logging
from typing import Any, Optional

from env.dig_env import Transition

logger = logging.getLogger(__name__)


class Agent:

    MAX_ACTIONS: int = 100_000

    action_counter: int = 0
    timer: float = 0.0
    game_id: str
    frames: list[Transition]

    def __init__(self, env: Any, *, game_id: str = "") -> None:
        self.env = env
        self.game_id = game_id or getattr(env, "game", "")
        self.frames: list[Transition] = []

    def open(self, *, session_id: Optional[str] = None) -> Transition:
        t = self.env.resume(session_id) if session_id else self.env.start()
        self.frames = [t]
        self.game_id = getattr(self.env, "game", self.game_id) or self.game_id
        return t

    def take_action(self, action: str, *, reasoning: Optional[str] = None) -> Optional[Transition]:
        try:
            return self.env.step(action, reasoning=reasoning)
        except Exception as e:
            logger.warning("env.step(%r) failed: %s: %s", action, type(e).__name__, e)
            return None

    def append_frame(self, frame: Transition) -> None:
        self.frames.append(frame)

    @property
    def latest(self) -> Transition:
        return self.frames[-1]

    @property
    def state(self):
        return self.frames[-1].state

    @property
    def levels_beaten(self) -> int:
        return int(self.frames[-1].levels_beaten)
