from __future__ import annotations

import logging
from typing import Optional

from arc_agi import EnvironmentWrapper
from arcengine import FrameData, FrameDataRaw, GameAction, GameState
from pydantic import ValidationError

logger = logging.getLogger(__name__)


class Agent:

    MAX_ACTIONS: int = 80

    action_counter: int = 0
    game_id: str
    frames: list[FrameData]
    arc_env: EnvironmentWrapper

    def __init__(self, *, arc_env: EnvironmentWrapper, game_id: str = "") -> None:
        self.game_id = game_id
        self.frames = [FrameData(levels_completed=0)]
        self.arc_env = arc_env

    @property
    def state(self) -> GameState:
        return self.frames[-1].state

    @property
    def levels_completed(self) -> int:
        return self.frames[-1].levels_completed

    def append_frame(self, frame: FrameData) -> None:
        self.frames.append(frame)

    def do_action_request(self, action: GameAction) -> FrameData:
        data = action.action_data.model_dump()
        reasoning = getattr(action, "reasoning", None)
        if reasoning is not None and not isinstance(reasoning, dict):
            reasoning = {"text": str(reasoning)}
        raw = self.arc_env.step(action, data=data, reasoning=reasoning)
        return self._convert_raw_frame_data(raw)

    def _convert_raw_frame_data(self, raw: FrameDataRaw | None) -> FrameData:
        if raw is None:
            raise ValueError("Received None frame data from environment")
        return FrameData(
            game_id=raw.game_id,
            frame=[arr.tolist() for arr in raw.frame],
            state=raw.state,
            levels_completed=raw.levels_completed,
            win_levels=raw.win_levels,
            guid=raw.guid,
            full_reset=raw.full_reset,
            available_actions=raw.available_actions,
        )

    def take_action(self, action: GameAction) -> Optional[FrameData]:
        frame_data = self.do_action_request(action)
        try:
            return FrameData.model_validate(frame_data)
        except ValidationError as e:
            logger.warning("Incoming frame data did not validate: %s", e)
            return None
