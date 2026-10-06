from __future__ import annotations

import threading
from typing import Optional

from rich.console import Console

from ..events import Event, EventSink

_ACTION_NAMES = {0: "RESET", 1: "↑", 2: "↓", 3: "←", 4: "→", 5: "act5", 6: "click", 7: "act7"}


def _args_summary(args: dict, cap: int = 96) -> str:
    parts = []
    for k, v in args.items():
        s = str(v).replace("\n", " ")
        if len(s) > 40:
            s = s[:37] + "…"
        parts.append(f"{k}={s}")
    out = ", ".join(parts)
    return out if len(out) <= cap else out[: cap - 1] + "…"


def _short(text: str, max_lines: int = 4, cap: int = 320) -> str:
    text = (text or "").rstrip()
    lines = text.split("\n")
    s = "\n".join(lines[:max_lines])
    if len(lines) > max_lines or len(s) > cap:
        s = s[:cap].rstrip() + " …"
    return s


def _flags(ev: Event) -> str:
    out = []
    if getattr(ev, "level_up", False):
        out.append("[bold green]LEVEL-UP[/]")
    if getattr(ev, "dead", False):
        out.append("[bold red]DEAD[/]")
    if getattr(ev, "win", False):
        out.append("[bold magenta]WIN[/]")
    return ("  " + " ".join(out)) if out else ""


class ConsoleSink(EventSink):
    def __init__(self, console: Optional[Console] = None) -> None:
        self.c = console or Console()
        self._lock = threading.Lock()
        self._stream: Optional[str] = None

    def emit(self, event: Event) -> None:
        with self._lock:
            self._dispatch(event)

    def _end_stream(self) -> None:
        if self._stream is not None:
            self.c.print()
            self._stream = None

    def _raw(self, text: str, **kw) -> None:
        self.c.print(text, end="", markup=False, highlight=False, soft_wrap=True, **kw)

    def _dispatch(self, ev: Event) -> None:
        k = ev.kind

        if k == "text_delta":
            if self._stream != "text":
                self._end_stream()
                self._stream = "text"
            self._raw(ev.text)
            return
        if k == "thinking_delta":
            if self._stream != "think":
                self._end_stream()
                self.c.print("[dim]💭 [/]", end="")
                self._stream = "think"
            self._raw(ev.text, style="dim italic")
            return

        self._end_stream()

        if k == "run_started":
            self.c.rule(
                f"[bold]world_model[/]  {ev.game_id}  ·  {ev.provider}"
                + (f"/{ev.model}" if ev.model else "")
                + f"  ·  ≤{ev.max_actions} actions",
                style="cyan",
            )
        elif k == "turn_started":
            self.c.print()
            self.c.rule(
                f"[bold cyan]turn {ev.turn}[/]  ·  step {ev.env_step}  ·  "
                f"{ev.state}  ·  level {ev.level}/{ev.win_levels}  ·  "
                f"legal {ev.legal}  ·  model {'✓' if ev.has_world_model else '—'}",
                style="dim cyan",
                align="left",
            )
            if ev.surprise:
                self.c.print(f"[yellow]⚠ {ev.surprise}[/]")
        elif k == "tool_started":
            self.c.print("  [cyan]⚙[/] ", end="")
            self._raw(ev.name, style="bold")
            summary = _args_summary(ev.args)
            if summary:
                self._raw("  " + summary, style="dim")
            self.c.print()
        elif k == "tool_finished":
            if ev.is_error:
                self.c.print("    [red]✗[/] ", end="")
                self._raw(_short(ev.output), style="red")
            else:
                self.c.print("    [green]✓[/] ", end="")
                self._raw(_short(ev.output), style="dim")
            self.c.print()
        elif k == "turn_committed":
            self.c.print(f"  [bold green]✔ commit[/] {ev.plan}", end="")
            if ev.reason:
                self._raw("  — " + ev.reason, style="dim")
            self.c.print()
        elif k == "turn_fallback":
            self.c.print(f"  [yellow]↩ fallback[/] [dim]{ev.reason}[/]")
        elif k == "action_taken":
            name = _ACTION_NAMES.get(ev.action, f"act{ev.action}")
            xy = f"@({ev.x},{ev.y})" if ev.x is not None else ""
            self.c.print(
                f"  [blue]▶[/] [{ev.step_index}] [bold]{name}[/]{xy}  "
                f"[dim]→ {ev.state} L{ev.level}[/]{_flags(ev)}"
            )
        elif k == "model_mispredicted":
            self.c.print(f"  [bold red]✗ misprediction[/] [dim]{ev.surprise}[/]")
        elif k == "run_finished":
            self.c.print()
            self.c.rule(
                f"[bold]done[/]  ·  {ev.state}  ·  level {ev.levels}/{ev.win_levels}  ·  "
                f"{ev.actions} actions  ·  {ev.transitions} transitions  ·  "
                f"model {'✓' if ev.has_world_model else '—'}",
                style="green",
            )

    def close(self) -> None:
        with self._lock:
            self._end_stream()
