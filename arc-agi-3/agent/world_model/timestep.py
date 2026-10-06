from __future__ import annotations

import numpy as np
from arcengine import FrameData, GameAction, GameState


class TimeStep:

    def __init__(
        self,
        *,
        action: GameAction,
        x: int | None,
        y: int | None,
        frame: FrameData,
        before_levels: int,
    ) -> None:
        self.action: GameAction = action
        self.x: int | None = None if x is None else int(x)
        self.y: int | None = None if y is None else int(y)
        self.frame: FrameData = frame
        self.before_levels: int = int(before_levels)

    @property
    def state(self) -> GameState:
        return self.frame.state

    @property
    def action_id(self) -> int:
        return int(self.action.value)

    @property
    def after(self) -> np.ndarray | None:
        if not self.frame.frame:
            return None
        return np.asarray(self.frame.frame[-1], dtype=np.int8)

    @property
    def ticks(self) -> list[np.ndarray]:
        return [np.asarray(g, dtype=np.int8) for g in self.frame.frame[:-1]]

    @property
    def levels_after(self) -> int:
        return int(self.frame.levels_completed)

    @property
    def level_up(self) -> bool:
        return self.frame.levels_completed > self.before_levels

    @property
    def dead(self) -> bool:
        return self.frame.state is GameState.GAME_OVER

    @property
    def win(self) -> bool:
        return self.frame.state is GameState.WIN


def encode_ticks(before: np.ndarray | None, frames: list) -> tuple[list | None, bool]:
    intermediate = frames[:-1] if frames else []
    if not intermediate:
        return None, False
    prev = None if before is None else np.asarray(before, dtype=int)
    out: list = []
    for grid in intermediate:
        current = np.asarray(grid, dtype=int)
        if prev is None or prev.shape != current.shape:
            out.append({"g": current.tolist()})
        else:
            ys, xs = np.where(prev != current)
            out.append({
                "d": [[int(x), int(y), int(current[y, x])]
                      for y, x in zip(ys.tolist(), xs.tolist())]
            })
        prev = current
    return out, False


def decode_ticks(before: np.ndarray | None, ticks: list | None) -> list[np.ndarray]:
    if not ticks:
        return []
    prev = None if before is None else np.asarray(before, dtype=int)
    out: list[np.ndarray] = []
    for tick in ticks:
        try:
            if "g" in tick:
                prev = np.asarray(tick["g"], dtype=int)
            elif prev is not None:
                current = np.array(prev, copy=True)
                for x, y, value in tick.get("d") or []:
                    current[int(y), int(x)] = int(value)
                prev = current
            else:
                continue
            out.append(prev.astype(np.int8))
        except Exception:
            continue
    return out
