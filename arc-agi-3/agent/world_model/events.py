from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np


def encode_grid(grid: Any) -> Optional[list[list[int]]]:
    if grid is None:
        return None
    arr = np.asarray(grid, dtype=int)
    if arr.ndim != 2:
        return None
    return arr.tolist()


@dataclass
class Event:
    kind: str = field(default="", init=False)
    seq: int = field(default=0, init=False)
    ts: float = field(default=0.0, init=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["kind"] = self.kind
        d["seq"] = self.seq
        d["ts"] = self.ts
        return d


@dataclass
class RunStarted(Event):
    kind = "run_started"
    game_id: str = ""
    provider: str = ""
    model: Optional[str] = None
    max_actions: int = 0
    win_levels: int = 0
    workdir: str = ""
    resumed: bool = False
    resumed_transitions: int = 0


@dataclass
class RunFinished(Event):
    kind = "run_finished"
    state: str = ""
    levels: int = 0
    win_levels: int = 0
    actions: int = 0
    transitions: int = 0
    has_world_model: bool = False


@dataclass
class TurnStarted(Event):
    kind = "turn_started"
    turn: int = 0
    env_step: int = 0
    state: str = ""
    level: int = 0
    win_levels: int = 0
    legal: list[int] = field(default_factory=list)
    grid: Optional[list[list[int]]] = None
    has_world_model: bool = False
    surprise: str = ""


@dataclass
class TextDelta(Event):
    kind = "text_delta"
    turn: int = 0
    text: str = ""


@dataclass
class ThinkingDelta(Event):
    kind = "thinking_delta"
    turn: int = 0
    text: str = ""


@dataclass
class ToolStarted(Event):
    kind = "tool_started"
    turn: int = 0
    call_id: str = ""
    name: str = ""
    args: dict = field(default_factory=dict)


@dataclass
class ToolFinished(Event):
    kind = "tool_finished"
    turn: int = 0
    call_id: str = ""
    name: str = ""
    output: str = ""
    is_error: bool = False


@dataclass
class TurnCommitted(Event):
    kind = "turn_committed"
    turn: int = 0
    plan: list[list[Any]] = field(default_factory=list)
    reason: str = ""


@dataclass
class TurnFallback(Event):
    kind = "turn_fallback"
    turn: int = 0
    reason: str = ""


@dataclass
class ActionTaken(Event):
    kind = "action_taken"
    turn: int = 0
    step_index: int = 0
    action: int = 0
    x: Optional[int] = None
    y: Optional[int] = None
    grid: Optional[list[list[int]]] = None
    level_up: bool = False
    dead: bool = False
    win: bool = False
    state: str = ""
    level: int = 0
    ticks: Optional[list] = None
    ticks_truncated: bool = False


@dataclass
class ModelMispredicted(Event):
    kind = "model_mispredicted"
    turn: int = 0
    step_index: int = 0
    surprise: str = ""
    predicted: Optional[list[list[int]]] = None
    actual: Optional[list[list[int]]] = None


class EventSink:

    def emit(self, event: Event) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class FanoutSink(EventSink):

    def __init__(self, sinks: list[EventSink], start_seq: int = 0) -> None:
        self._sinks = list(sinks)
        self._seq = int(start_seq)
        self._lock = threading.Lock()

    def emit(self, event: Event) -> None:
        with self._lock:
            self._seq += 1
            event.seq = self._seq
            event.ts = time.time()
        for sink in self._sinks:
            try:
                sink.emit(event)
            except Exception:
                pass

    def close(self) -> None:
        for sink in self._sinks:
            try:
                sink.close()
            except Exception:
                pass
