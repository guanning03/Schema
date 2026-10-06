from __future__ import annotations

from typing import Callable

from .events import EventSink, TextDelta, ThinkingDelta


def make_on_event(provider: str, sink: EventSink, turn: int) -> Callable[[dict], None]:
    if provider == "codex-cli":
        return _make_codex(sink, turn)
    return _make_claude(sink, turn)


def _make_claude(sink: EventSink, turn: int) -> Callable[[dict], None]:

    def on_event(msg: dict) -> None:
        if msg.get("type") != "stream_event":
            return
        event = msg.get("event") or {}
        if event.get("type") != "content_block_delta":
            return
        delta = event.get("delta") or {}
        dtype = delta.get("type")
        if dtype == "text_delta":
            text = delta.get("text") or ""
            if text:
                sink.emit(TextDelta(turn=turn, text=text))
        elif dtype == "thinking_delta":
            text = delta.get("thinking") or ""
            if text:
                sink.emit(ThinkingDelta(turn=turn, text=text))

    return on_event


def _make_codex(sink: EventSink, turn: int) -> Callable[[dict], None]:

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
