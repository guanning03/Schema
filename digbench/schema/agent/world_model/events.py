from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Optional


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
    max_level: Optional[int] = None
    workdir: str = ""
    resumed: bool = False
    resumed_transitions: int = 0
    session_id: Optional[str] = None
    base_url: str = ""
    model_name: Optional[str] = None
    description: Optional[str] = None
    seed: Optional[int] = None
    framework_version: Optional[str] = None
    initial_state: Optional[dict] = None
    initial_step_index: int = 0
    initial_levels_beaten: int = 0


@dataclass
class RunFinished(Event):
    kind = "run_finished"
    status: str = ""
    level: int = 0
    max_level: Optional[int] = None
    levels_beaten: int = 0
    actions: int = 0
    transitions: int = 0
    has_world_model: bool = False
    stop_reason: str = ""


@dataclass
class TurnStarted(Event):

    kind = "turn_started"
    turn: int = 0
    env_step: int = 0
    status: str = ""
    level: int = 0
    max_level: Optional[int] = None
    lives_left: Optional[int] = None
    steps_remaining: Optional[int] = None
    max_steps: Optional[int] = None
    mode: Optional[str] = None
    transition: Optional[str] = None
    legal: list[str] = field(default_factory=list)
    observation: Optional[str] = None
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
    plan: list[str] = field(default_factory=list)
    reason: str = ""
    usage: dict = field(default_factory=dict)


@dataclass
class TurnFallback(Event):

    kind = "turn_fallback"
    turn: int = 0
    reason: str = ""
    usage: dict = field(default_factory=dict)


@dataclass
class ActionTaken(Event):

    kind = "action_taken"
    turn: int = 0
    step_index: int = 0
    server_step_index: int = 0
    action: str = ""
    observation: Optional[str] = None
    legal: list[str] = field(default_factory=list)
    level: int = 0
    max_level: Optional[int] = None
    lives_left: Optional[int] = None
    steps_remaining: Optional[int] = None
    max_steps: Optional[int] = None
    mode: Optional[str] = None
    status: str = ""
    done: bool = False
    transition: Optional[str] = None
    invalid: bool = False
    levels_beaten: int = 0
    events: list = field(default_factory=list)
    level_up: bool = False
    life_lost: bool = False
    dead: bool = False
    win: bool = False
    mode_switch: bool = False
    budget_exhausted: bool = False
    segment: int = 0
    state: Optional[dict] = None


@dataclass
class ModelMispredicted(Event):

    kind = "model_mispredicted"
    turn: int = 0
    step_index: int = 0
    surprise: str = ""
    predicted: Optional[str] = None
    actual: Optional[str] = None
    predicted_flags: Optional[dict] = None
    actual_flags: Optional[dict] = None


@dataclass
class RunControl(Event):

    kind = "run_control"
    turn: int = 0
    state: str = ""
    ref: str = ""
    detail: str = ""


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
