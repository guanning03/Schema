from __future__ import annotations

from typing import Callable

from .events import EventSink, TextDelta, ThinkingDelta


def make_on_event(sink: EventSink, turn: int) -> Callable[[dict], None]:

    def on_event(ev: dict) -> None:
        etype = ev.get("type", "")
        if etype == "response.output_text.delta":
            text = ev.get("delta") or ""
            if text:
                sink.emit(TextDelta(turn=turn, text=text))
        elif etype in (
            "response.reasoning_summary_text.delta",
            "response.reasoning_text.delta",
        ):
            text = ev.get("delta") or ""
            if text:
                sink.emit(ThinkingDelta(turn=turn, text=text))

    return on_event
