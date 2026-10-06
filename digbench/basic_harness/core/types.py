from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class Move:

    action: str | None
    reasoning_summary: str
    has_continuity: bool
    finish_reason: str | None
    usage: dict
    cost: float | None
    elapsed_s: float
    error: dict | None = None
    thoughts_basis: str = "none"
    continuity: str | None = None


@dataclass
class Turn:

    payload: object
    is_active: bool
    est_tokens: int


@runtime_checkable
class Policy(Protocol):

    model: str
    model_max_context: int | None
    has_pricing: bool
    last_prompt_tokens: int

    def start(self, description: str, state_slice: dict) -> None: ...
    def generate_move(self) -> Move: ...
    def observe(self, result: dict) -> None: ...
    def add_nudge(self, text: str) -> None: ...
    def debrief(self) -> str | None: ...

    def turns(self) -> list[Turn]: ...
    def evict_oldest_turn(self) -> None: ...
