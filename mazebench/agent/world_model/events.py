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
    workdir: str = ""
    resumed: bool = False
    resumed_transitions: int = 0
    engine_root: str = ""
    engine_commit: str = ""
    start_room: str = ""
    view: str = ""
    yaw: int = 0
    hide_names: bool = True
    hide_names_seed: str = "1"
    initial_state: Optional[dict] = None


@dataclass
class RunFinished(Event):
    kind = "run_finished"
    status: str = ""
    gems: int = 0
    rooms_visited: int = 0
    actions: int = 0
    transitions: int = 0
    has_world_model: bool = False
    stop_reason: str = ""
    scorecard: Optional[dict] = None


@dataclass
class ScorecardUpdated(Event):
    kind = "scorecard"
    scorecard: Optional[dict] = None


@dataclass
class TurnStarted(Event):

    kind = "turn_started"
    turn: int = 0
    env_step: int = 0
    room: str = ""
    view: str = ""
    yaw: int = 0
    gems: int = 0
    rooms_visited: int = 0
    transition: Optional[str] = None
    legal: list[str] = field(default_factory=list)
    observation: Optional[str] = None
    has_world_model: bool = False
    surprise: str = ""


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
class TurnUsage(Event):

    kind = "turn_usage"
    turn: int = 0
    input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    output_tokens: int = 0
    raw: dict = field(default_factory=dict)


@dataclass
class TurnCommitted(Event):

    kind = "turn_committed"
    turn: int = 0
    plan: list[str] = field(default_factory=list)
    reason: str = ""
    usage: dict = field(default_factory=dict)
    suggestion: str = ""


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
    env_step_index: int = 0
    action: str = ""
    observation: Optional[str] = None
    legal: list[str] = field(default_factory=list)
    room: str = ""
    view: str = ""
    yaw: int = 0
    gems: int = 0
    rooms_visited: int = 0
    done: bool = False
    transition: Optional[str] = None
    invalid: bool = False
    gem: bool = False
    room_changed: bool = False
    dead: bool = False
    won: bool = False
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
