from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.request

from . import clientutil


USER_AGENT = "digbench-baseline-harness/1.0 (+https://digbench.ai)"


class BenchError(RuntimeError):
    pass


class Bench:

    def __init__(self, base: str, token: str, *, timeout: int, max_retries: int):
        self.base = base
        self.token = token
        self.timeout = timeout
        self.max_retries = max_retries
        self.on_retry = None

    def _call(self, method: str, path: str, payload: dict | None = None) -> dict:
        data = json.dumps(payload).encode() if payload is not None else None
        last = None
        for attempt in range(self.max_retries):
            req = urllib.request.Request(
                self.base + path,
                data=data,
                method=method,
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": USER_AGENT,
                },
            )
            try:
                with clientutil.urlopen_no_redirect(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", "replace")
                if exc.code < 500 and exc.code not in (408, 429):
                    raise BenchError(f"{method} {path} -> HTTP {exc.code}: {body}")
                last = f"HTTP {exc.code}: {body}"
            except urllib.error.URLError as exc:
                last = f"network: {exc.reason}"
            except (OSError, http.client.HTTPException, ValueError) as exc:
                last = f"read/parse: {exc}"
            if attempt < self.max_retries - 1:
                if self.on_retry:
                    self.on_retry(
                        f"bench {method} {path} attempt {attempt + 1}/{self.max_retries} "
                        f"failed: {last}; retrying"
                    )
                time.sleep(min(2 ** attempt, 30))
        raise BenchError(f"{method} {path} failed after {self.max_retries} attempts: {last}")

    def start_session(self, game: str, model_name: str, model_version: str) -> dict:
        return self._call(
            "POST", "/sessions",
            {"game": game, "model_name": model_name, "model_version": model_version},
        )

    def step(self, sid: str, step_index: int, action: str) -> dict:
        return self._call(
            "POST", f"/sessions/{sid}/step", {"step_index": step_index, "action": action}
        )


def state_for_model(state: dict) -> dict:
    out = {
        "observation": state.get("observation", ""),
        "level": state.get("level"),
        "max_level": state.get("max_level"),
        "lives_left": state.get("lives_left"),
        "steps_remaining": state.get("steps_remaining"),
        "status": state.get("status"),
        "done": state.get("done"),
        "legal_actions": state.get("actions", []),
    }
    if state.get("mode") is not None:
        out["mode"] = state["mode"]
        out["creative_toggle"] = state.get("creative_toggle")
    if state.get("transition") is not None:
        out["transition"] = state["transition"]
    return out


def fmt_level(state: dict) -> str:
    level, mx = state.get("level"), state.get("max_level")
    return f"{level}/{mx}" if mx is not None else str(level)


def levels_beaten(state: dict) -> int | None:
    level = state.get("level")
    if not isinstance(level, int):
        return None
    if state.get("status") == "completed":
        mx = state.get("max_level")
        if isinstance(mx, int):
            return mx
    return max(0, level - 1)


def terminal_banner(state: dict) -> str:
    status = state.get("status")
    if status == "completed":
        return "Game completed!"
    if status == "game_over":
        return "Game over — out of lives" if state.get("lives_left") == 0 else "Game over"
    return f"Game ended ({status})"
