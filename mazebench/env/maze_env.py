from __future__ import annotations

import json
import os
import re
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_REPO = Path(__file__).resolve().parent.parent

_LEVEL_RE = re.compile(r"^(?:level_)?([A-Z])x([A-Z])$")
_GOTO_RE = re.compile(r"^(?:go\s+to\s+level|goto)\s+([A-Za-z])\s*[x ]?\s*([A-Za-z])$", re.I)
_ROTATE_RE = re.compile(r"^rotate(?:\s+camera)?\s+(up|down|left|right)$", re.I)
_MOVES = ("up", "down", "left", "right")

VIEW_NAMES = ("top", "top-diagonal", "diagonal", "side-diagonal", "side")
GAME_WON_GEM_COUNT = 100


class MazeError(RuntimeError):
    pass


def engine_root() -> Path:
    for cand in (os.environ.get("MAZEBENCH_ENGINE_ROOT"),
                 _REPO.parent / "vendor" / "MazeBenchEngine", _REPO / "vendor" / "MazeBenchEngine"):
        if not cand:
            continue
        p = Path(cand).expanduser()
        if (p / "scripts" / "maze-bridge.js").is_file():
            return p.resolve()
    raise MazeError(
        "MazeBenchEngine not found (needs scripts/maze-bridge.js). "
        "Set MAZEBENCH_ENGINE_ROOT, or put the engine at vendor/MazeBenchEngine.")


def node_bin() -> str:
    cand = os.environ.get("MAZEBENCH_NODE_BIN")
    if cand:
        return cand
    vendored = _REPO.parent / "vendor" / "node" / "bin" / "node"
    return str(vendored) if vendored.is_file() else "node"


def normalize_room(value: str) -> str:
    m = _LEVEL_RE.match(str(value or "").strip())
    return f"level_{m.group(1)}x{m.group(2)}" if m else str(value or "").strip()


def parse_action(action: str) -> dict:
    text = " ".join(str(action or "").strip().lower().split())
    if text in _MOVES:
        return {"command": "move", "direction": text}
    m = re.match(r"^move\s+(up|down|left|right)$", text)
    if m:
        return {"command": "move", "direction": m.group(1)}
    m = _ROTATE_RE.match(text)
    if m:
        return {"command": "rotate_camera", "direction": m.group(1).lower()}
    if text == "undo":
        return {"command": "undo"}
    if text in ("reset", "reset level"):
        return {"command": "reset_level"}
    if text == "quit":
        return {"command": "quit"}
    m = _GOTO_RE.match(text)
    if m:
        return {"command": "goto_level", "x": m.group(1).upper(), "y": m.group(2).upper()}
    raise ValueError(
        f"unknown action {action!r} — expected one of: up, down, left, right, "
        "rotate camera up|down|left|right, undo, reset, go to level X Y, quit")


def canonical_action(action: str) -> str:
    msg = parse_action(action)
    c = msg["command"]
    if c == "move":
        return msg["direction"]
    if c == "rotate_camera":
        return f"rotate camera {msg['direction']}"
    if c == "reset_level":
        return "reset"
    if c == "goto_level":
        return f"go to level {msg['x']} {msg['y']}"
    return c


DEAD_ACTIONS = ("undo", "reset", "go to level X Y")


@dataclass
class MazeState:

    observation: str = ""
    actions: list = field(default_factory=list)
    room: str = ""
    view: str = "top-diagonal"
    yaw: int = 0
    gems: int = 0
    visited_rooms: list = field(default_factory=list)
    dead: bool = False
    death_message: str = ""
    won: bool = False
    lost: bool = False
    action_count: int = 0
    transition: Optional[str] = None

    @property
    def done(self) -> bool:
        return self.won or self.lost

    def summary(self) -> str:
        return (f"room {self.room} | view {self.view} yaw {self.yaw} | "
                f"gems {self.gems}/{GAME_WON_GEM_COUNT} | rooms visited {len(self.visited_rooms)}"
                + (" | DEAD" if self.dead else ""))

    def to_dict(self) -> dict:
        return {"observation": self.observation, "actions": list(self.actions), "room": self.room,
                "view": self.view, "yaw": self.yaw, "gems": self.gems,
                "visited_rooms": list(self.visited_rooms), "dead": self.dead,
                "death_message": self.death_message, "won": self.won, "lost": self.lost,
                "action_count": self.action_count, "transition": self.transition}

    @classmethod
    def from_dict(cls, d: dict) -> "MazeState":
        d = dict(d or {})
        return cls(observation=str(d.get("observation") or ""),
                   actions=[str(a) for a in (d.get("actions") or [])],
                   room=str(d.get("room") or ""), view=str(d.get("view") or "top-diagonal"),
                   yaw=int(d.get("yaw") or 0), gems=int(d.get("gems") or 0),
                   visited_rooms=[str(r) for r in (d.get("visited_rooms") or [])],
                   dead=bool(d.get("dead")), death_message=str(d.get("death_message") or ""),
                   won=bool(d.get("won")), lost=bool(d.get("lost")),
                   action_count=int(d.get("action_count") or 0),
                   transition=(str(d["transition"]) if d.get("transition") else None))


@dataclass
class Transition:

    state: MazeState
    step_index: int
    action: Optional[str] = None
    invalid_action: bool = False
    gem_collected: int = 0
    room_changed: bool = False
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def observation(self) -> str:
        return self.state.observation

    @property
    def legal_actions(self) -> list:
        return list(self.state.actions)

    @property
    def done(self) -> bool:
        return self.state.done

    def __repr__(self) -> str:
        return (f"Transition(step={self.step_index} action={self.action!r} "
                f"{self.state.summary()} gem+{self.gem_collected})")


def _transition_note(*, gem: int, room_changed: bool, dead: bool, won: bool,
                     action: Optional[str]) -> Optional[str]:
    bits = []
    if gem > 0:
        bits.append(f"Collected {gem} gem{'s' if gem > 1 else ''}.")
    if room_changed and action is not None:
        bits.append("You are now in a different room.")
    if dead:
        bits.append("The player died.")
    if won:
        bits.append("Game won.")
    return " ".join(bits) or None


class MazeEnv:

    def __init__(self, *, room: str, view: str, yaw: int, hide_names_seed: str) -> None:
        self.engine = engine_root()
        self.node = node_bin()
        self.room = normalize_room(room)
        self.view = view if view in VIEW_NAMES else "top-diagonal"
        self.yaw = int(yaw) % 4
        self.hide_names = True
        self.hide_names_seed = str(hide_names_seed)
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._step_index = 0
        self._gems = 0
        self._room_now = self.room

    def _argv(self) -> list:
        argv = [self.node, str(self.engine / "scripts" / "maze-bridge.js"),
                "--level", self.room, "--view", self.view, "--yaw", str(self.yaw),
                "--observation-mode", "text"]
        argv += ["--hide-names", "--hide-names-seed", self.hide_names_seed]
        return argv

    def _ensure(self) -> subprocess.Popen:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        env = dict(os.environ, MAZEBENCH_REPO_ROOT=str(self.engine))
        self._proc = subprocess.Popen(
            self._argv(), cwd=str(self.engine), env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            bufsize=1)
        return self._proc

    def _rpc(self, message: dict) -> dict:
        with self._lock:
            proc = self._ensure()
            try:
                proc.stdin.write(json.dumps(message) + "\n")
                proc.stdin.flush()
            except (BrokenPipeError, ValueError) as e:
                raise MazeError(f"bridge exited ({e}); command {message.get('command')}") from e
            line = proc.stdout.readline()
        if not line:
            err = ""
            try:
                err = (self._proc.stderr.read() or "")[-500:] if self._proc else ""
            except Exception:
                pass
            raise MazeError(f"bridge returned nothing ({message.get('command')}){' — ' + err if err else ''}")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as e:
            raise MazeError(f"bridge returned non-JSON: {line[:300]!r}") from e
        if payload.get("error"):
            raise MazeError(str(payload["error"]))
        return payload

    def _state(self, snap: dict, *, action: Optional[str], gem: int,
               room_changed: bool) -> MazeState:
        dead = bool(snap.get("player_dead"))
        gems = int(snap.get("gem_count") or 0)
        won = bool(snap.get("game_won")) or gems >= GAME_WON_GEM_COUNT
        lost = bool(snap.get("game_lost"))
        allowed = [str(a) for a in (snap.get("allowed_commands") or [])]
        if dead and not allowed:
            allowed = list(DEAD_ACTIONS)
        allowed = [a for a in allowed if a.strip().lower() != "quit"]
        return MazeState(
            observation=str(snap.get("level") or ""),
            actions=allowed,
            room=str(snap.get("current_room") or ""),
            view=str(snap.get("current_view") or self.view),
            yaw=int(snap.get("yaw") or 0),
            gems=gems,
            visited_rooms=[str(r) for r in (snap.get("visited_levels") or [])],
            dead=dead,
            death_message=str(snap.get("death_message") or "") if dead else "",
            won=won, lost=lost,
            action_count=int(snap.get("action_count") or 0),
            transition=_transition_note(gem=gem, room_changed=room_changed, dead=dead,
                                        won=won, action=action),
        )

    def _wrap(self, snap: dict, *, action: Optional[str], consumed: bool) -> Transition:
        gems = int(snap.get("gem_count") or 0)
        gem = max(0, gems - self._gems)
        room = str(snap.get("current_room") or "")
        room_changed = bool(room and room != self._room_now)
        self._gems, self._room_now = gems, room or self._room_now
        if consumed:
            self._step_index += 1
        state = self._state(snap, action=action, gem=gem, room_changed=room_changed)
        return Transition(state=state, step_index=self._step_index, action=action,
                          gem_collected=gem, room_changed=room_changed, raw=snap)

    def start(self) -> Transition:
        snap = self._rpc({"command": "observe"})
        self._gems = int(snap.get("gem_count") or 0)
        self._room_now = str(snap.get("current_room") or self.room)
        return self._wrap(snap, action=None, consumed=False)

    def step(self, action: str) -> Transition:
        msg = parse_action(action)
        canon = canonical_action(action)
        if canon == "quit":
            cur = self._rpc({"command": "observe"})
            t = self._wrap(cur, action=canon, consumed=False)
            t.invalid_action = True
            t.state.transition = "That action was refused: quit is disabled for this run."
            return t
        try:
            snap = self._rpc(msg)
        except MazeError as e:
            cur = self._rpc({"command": "observe"})
            t = self._wrap(cur, action=canon, consumed=False)
            t.invalid_action = True
            t.state.transition = f"That action was refused: {e}"
            return t
        return self._wrap(snap, action=canon, consumed=True)

    def replay(self, actions: "list") -> "Transition":
        self._rpc({"command": "replay_fast", "enabled": True})
        try:
            for a in actions:
                msg = parse_action(a)
                if msg["command"] == "quit":
                    continue
                try:
                    self._rpc(msg)
                    self._step_index += 1
                except MazeError:
                    pass
        finally:
            self._rpc({"command": "replay_fast", "enabled": False})
        snap = self._rpc({"command": "observe"})
        self._gems = int(snap.get("gem_count") or 0)
        self._room_now = str(snap.get("current_room") or self._room_now)
        return self._wrap(snap, action=None, consumed=False)

    def scorecard(self) -> dict:
        snap = self._rpc({"command": "scorecard"})
        return dict(snap.get("scorecard") or {})

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.stdin.write(json.dumps({"command": "close"}) + "\n")
            proc.stdin.flush()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
