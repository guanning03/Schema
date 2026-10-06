from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

import httpx

BASE_URL = "https://chatgpt.com/backend-api/codex/responses"
_TOKEN_URL = "https://auth.openai.com/oauth/token"
_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
_ORIGINATOR = "codex_cli_rs"
_CLIENT_VERSION = "0.153.4"
_AUTH_PATH = Path(os.path.expanduser("~/.codex/auth.json"))

_POOL_ENV = "ARC_CODEX_POOL"
_POOL_GLOB = ".codex-arc-agent*"
_FALLBACK_LIMIT_COOLDOWN = 300.0
_COOLDOWN_CAP = 1800.0
LIMIT_MSG_RE = re.compile(r"usage.?limit|rate.?limit|quota|too many requests", re.IGNORECASE)


class CodexAuthError(RuntimeError):
    pass


class CodexLimitError(RuntimeError):
    def __init__(self, kind: str, reset_at: Optional[float], detail: str = "") -> None:
        super().__init__(detail or kind)
        self.kind = kind
        self.reset_at = reset_at


def find_reset_seconds(obj) -> Optional[float]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("resets_in_seconds", "reset_after_seconds", "retry_after") and isinstance(v, (int, float)):
                return float(v)
            got = find_reset_seconds(v)
            if got is not None:
                return got
    elif isinstance(obj, list):
        for v in obj:
            got = find_reset_seconds(v)
            if got is not None:
                return got
    return None


def classify_limit(status: int, headers, body_text: str) -> "Optional[tuple[str, Optional[float]]]":
    if status != 429:
        return None
    secs: Optional[float] = None
    try:
        secs = find_reset_seconds(json.loads(body_text))
    except Exception:
        pass
    if secs is None:
        for h in ("retry-after", "x-codex-primary-reset-after-seconds"):
            try:
                secs = float(headers.get(h, ""))
                break
            except (TypeError, ValueError):
                continue
    return "limit", (time.time() + secs if secs is not None else None)


class CodexAccountPool:

    def __init__(self, dirs: list[str], notify: Optional[Callable[[str], None]] = None) -> None:
        self._dirs = list(dirs)
        self._cooldown: dict[str, float] = {}
        self._idx = 0
        self._lock = threading.Lock()
        self._notify = notify or (lambda m: print(m, file=sys.stderr, flush=True))

    def __len__(self) -> int:
        return len(self._dirs)

    @classmethod
    def discover(cls, notify: Optional[Callable[[str], None]] = None) -> "Optional[CodexAccountPool]":
        raw = os.environ.get(_POOL_ENV, "").strip()
        if raw:
            dirs = [os.path.expanduser(p) for p in re.split(r"[:,]", raw) if p.strip()]
        else:
            dirs = sorted(str(p) for p in Path.home().glob(_POOL_GLOB) if p.is_dir())
        dirs = [d for d in dirs if os.path.isfile(os.path.join(d, "auth.json"))]
        if not dirs:
            return None
        return cls(dirs, notify=notify)

    def acquire(self) -> str:
        while True:
            now = time.time()
            with self._lock:
                for d in list(self._cooldown):
                    if self._cooldown[d] <= now:
                        del self._cooldown[d]
                n = len(self._dirs)
                for off in range(n):
                    d = self._dirs[(self._idx + off) % n]
                    if d not in self._cooldown:
                        self._idx = self._dirs.index(d)
                        return d
                wake = min(self._cooldown.values())
            sleep_for = max(1.0, wake - time.time())
            self._notify(
                f"[codex-pool] all {len(self._dirs)} accounts are cooling down; sleeping "
                f"{int(sleep_for)}s until the earliest reset "
                f"({time.strftime('%H:%M:%S', time.localtime(wake))})…"
            )
            time.sleep(min(sleep_for, 60.0))

    def mark_limited(self, account_dir: str, kind: str, reset_at: Optional[float]) -> None:
        if reset_at is None:
            reset_at = time.time() + _FALLBACK_LIMIT_COOLDOWN
        reset_at = min(reset_at, time.time() + _COOLDOWN_CAP)
        with self._lock:
            self._cooldown[account_dir] = reset_at
            self._idx = (self._idx + 1) % len(self._dirs)
        when = time.strftime("%H:%M:%S", time.localtime(reset_at))
        self._notify(f"[codex-pool] {account_dir} hit {kind} (expected reset {when}), switching account.")


class CodexAuth:

    def __init__(self, *, timeout: float) -> None:
        self.timeout = timeout
        self._session_uuid = uuid.uuid4().hex
        self._notify = lambda m: print(m, file=sys.stderr, flush=True)
        self.pool = CodexAccountPool.discover(self._notify)
        if self.pool is not None:
            self._auth_path = Path(self.pool.acquire()) / "auth.json"
            self._notify(
                f"[codex-pool] {len(self.pool)} account(s), using {self._auth_path.parent.name}"
            )
        else:
            self._auth_path = _AUTH_PATH
        self._auth = self._load_auth()

    def available(self) -> bool:
        return bool(self._auth.get("access_token"))

    def _load_auth(self) -> dict:
        try:
            data = json.loads(self._auth_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        tokens = data.get("tokens") or {}
        return {
            "access_token": tokens.get("access_token") or data.get("access_token"),
            "refresh_token": tokens.get("refresh_token") or data.get("refresh_token"),
            "account_id": tokens.get("account_id") or data.get("account_id"),
        }

    def rotate_account(self, kind: str, reset_at: Optional[float]) -> None:
        assert self.pool is not None
        self.pool.mark_limited(str(self._auth_path.parent), kind, reset_at)
        self._auth_path = Path(self.pool.acquire()) / "auth.json"
        self._auth = self._load_auth()

    def refresh(self) -> bool:
        try:
            fresh = self._load_auth()
            if fresh.get("access_token") and fresh.get("access_token") != self._auth.get("access_token"):
                self._auth = fresh
                return True
            if fresh.get("refresh_token"):
                self._auth["refresh_token"] = fresh["refresh_token"]
        except Exception:
            pass
        rt = self._auth.get("refresh_token")
        if not rt:
            return False
        try:
            r = httpx.post(
                _TOKEN_URL,
                data={"grant_type": "refresh_token", "refresh_token": rt, "client_id": _CLIENT_ID},
                timeout=60.0,
            )
            r.raise_for_status()
            tok = r.json()
        except Exception:
            return False
        self._auth["access_token"] = tok.get("access_token", self._auth.get("access_token"))
        if tok.get("refresh_token"):
            self._auth["refresh_token"] = tok["refresh_token"]
        try:
            data = json.loads(self._auth_path.read_text(encoding="utf-8"))
            data.setdefault("tokens", {})
            data["tokens"]["access_token"] = self._auth["access_token"]
            data["tokens"]["refresh_token"] = self._auth["refresh_token"]
            data["last_refresh"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self._auth_path.write_text(json.dumps(data), encoding="utf-8")
        except Exception:
            pass
        return True

    def headers(self) -> dict:
        h = {
            "Authorization": f"Bearer {self._auth.get('access_token', '')}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "originator": _ORIGINATOR,
            "version": _CLIENT_VERSION,
        }
        if self._auth.get("account_id"):
            h["ChatGPT-Account-ID"] = self._auth["account_id"]
        return h
