from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

DEFAULT_SERVER = "https://api.digbench.ai"
API_PREFIX = "/api/agent"
USER_AGENT = "schema-digbench-harness/0.1 (+https://digbench.ai)"

_TRANSIENT_HTTP = {408, 429, 500, 502, 503, 504}


class DigBenchError(RuntimeError):
    pass


class DigBenchAuthError(DigBenchError):
    pass


class DigBenchProtocolError(DigBenchError):
    pass


class _Transient(Exception):

    def __init__(self, msg: str, retry_after: Optional[str] = None) -> None:
        super().__init__(msg)
        self.retry_after = retry_after


def server_base_url() -> str:
    return DEFAULT_SERVER + API_PREFIX


def api_token() -> str:
    return os.environ.get("DIGBENCH_API_TOKEN", "").strip()


@dataclass
class DigState:

    observation: str = ""
    actions: list[str] = field(default_factory=list)
    level: int = 1
    max_level: Optional[int] = None
    lives_left: Optional[int] = None
    starting_lives: Optional[int] = None
    steps_remaining: Optional[int] = None
    max_steps: Optional[int] = None
    mode: Optional[str] = None
    creative_toggle: Optional[str] = None
    creative_toggle_available: Optional[bool] = None
    creative_unavailable_reason: Optional[str] = None
    status: str = "in_progress"
    done: bool = False
    transition: Optional[str] = None
    extra: dict = field(default_factory=dict)

    _KNOWN = (
        "observation", "actions", "level", "max_level", "lives_left", "starting_lives",
        "steps_remaining", "max_steps", "mode", "creative_toggle", "creative_toggle_available",
        "creative_unavailable_reason", "status", "done", "transition",
    )

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "DigState":
        d = dict(d or {})
        acts = d.get("actions") or []
        st = cls(
            observation=str(d.get("observation") if d.get("observation") is not None else ""),
            actions=[str(a) for a in acts],
            level=_int_or(d.get("level"), 1),
            max_level=_opt_int(d.get("max_level")),
            lives_left=_opt_int(d.get("lives_left")),
            starting_lives=_opt_int(d.get("starting_lives")),
            steps_remaining=_opt_int(d.get("steps_remaining")),
            max_steps=_opt_int(d.get("max_steps")),
            mode=(str(d["mode"]) if d.get("mode") is not None else None),
            creative_toggle=(str(d["creative_toggle"]) if d.get("creative_toggle") is not None else None),
            creative_toggle_available=(None if d.get("creative_toggle_available") is None
                                       else bool(d.get("creative_toggle_available"))),
            creative_unavailable_reason=(str(d["creative_unavailable_reason"])
                                         if d.get("creative_unavailable_reason") is not None else None),
            status=str(d.get("status") or "in_progress"),
            done=bool(d.get("done", False)),
            transition=(str(d["transition"]) if d.get("transition") is not None else None),
        )
        st.extra = {k: v for k, v in d.items() if k not in cls._KNOWN}
        return st

    def to_dict(self) -> dict:
        out = {k: getattr(self, k) for k in self._KNOWN}
        out["actions"] = list(self.actions)
        if self.extra:
            out["extra"] = dict(self.extra)
        return out

    @property
    def won(self) -> bool:
        return self.status == "completed"

    @property
    def game_over(self) -> bool:
        return self.status == "game_over"

    @property
    def in_creative(self) -> bool:
        return (self.mode or "").lower() == "creative"

    def summary(self) -> str:
        parts = [f"status={self.status}", f"level {self.level}/{self.max_level if self.max_level is not None else '?'}"]
        if self.lives_left is not None:
            parts.append(f"lives {self.lives_left}" + (f"/{self.starting_lives}" if self.starting_lives is not None else ""))
        if self.steps_remaining is not None:
            parts.append(f"steps_remaining {self.steps_remaining}" + (f"/{self.max_steps}" if self.max_steps is not None else ""))
        if self.mode is not None:
            parts.append(f"mode {self.mode}")
        return " | ".join(parts)


@dataclass
class Transition:

    state: DigState
    step_index: int
    action: Optional[str] = None
    invalid_action: bool = False
    events: list[dict] = field(default_factory=list)
    levels_beaten: int = 0
    done: bool = False
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def observation(self) -> str:
        return self.state.observation

    @property
    def won(self) -> bool:
        return self.state.won

    @property
    def game_over(self) -> bool:
        return self.state.game_over

    @classmethod
    def from_response(cls, d: dict, *, action: Optional[str] = None) -> "Transition":
        st = DigState.from_dict(d.get("state"))
        return cls(
            state=st,
            step_index=_int_or(d.get("step_index"), 0),
            action=action,
            invalid_action=bool(d.get("invalid_action") or False),
            events=[e for e in (d.get("events") or []) if isinstance(e, dict)],
            levels_beaten=_int_or(d.get("levels_beaten"), max(0, st.level - 1)),
            done=bool(d.get("done", st.done)),
            raw=d,
        )

    def __repr__(self) -> str:
        return (f"Transition(step={self.step_index} action={self.action!r} {self.state.summary()} "
                f"done={self.done} invalid={self.invalid_action} obs={self.state.observation!r})")


def _int_or(v: Any, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _parse_ts(v: Any) -> Optional[float]:
    if not isinstance(v, str) or not v:
        return None
    try:
        from datetime import datetime, timezone

        dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return None


def _opt_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


class _Http:

    def __init__(self, base: str, token: str, *, timeout: float, user_agent: str) -> None:
        self.base = base
        self.timeout = timeout
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": user_agent,
        }
        self._client = None
        try:
            import httpx

            self._client = httpx.Client(timeout=timeout, trust_env=True, follow_redirects=False)
        except Exception:
            self._client = None

    def request(self, method: str, path: str, payload: Optional[dict],
                *, timeout: Optional[float] = None) -> "tuple[int, dict, dict]":
        url = self.base + path
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        timeout = self.timeout if timeout is None else timeout
        if self._client is not None:
            r = self._client.request(method, url, content=data, headers=self.headers, timeout=timeout)
            body = _parse_json(r.content)
            return r.status_code, {k.lower(): v for k, v in r.headers.items()}, body
        import urllib.error
        import urllib.request

        req = urllib.request.Request(url, data=data, method=method, headers=self.headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, {k.lower(): v for k, v in resp.headers.items()}, _parse_json(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, {k.lower(): v for k, v in e.headers.items()}, _parse_json(e.read())

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass


def _parse_json(raw: bytes) -> dict:
    if not raw:
        return {}
    try:
        v = json.loads(raw.decode("utf-8", "replace"))
    except ValueError:
        return {"_raw": raw.decode("utf-8", "replace")[:2000]}
    return v if isinstance(v, dict) else {"_value": v}


class DigEnv:

    def __init__(
        self,
        game: str,
        *,
        model_name: Optional[str],
        model_version: Optional[str] = None,
        notes: Optional[dict | str] = None,
        timeout: float = 60.0,
        session_timeout: float = 180.0,
        max_retries: int = 8,
        on_retry=None,
    ) -> None:
        self.game = str(game)
        self.base_url = server_base_url()
        self._token = api_token()
        if not self._token:
            raise DigBenchAuthError(
                "no DigBench API token: export DIGBENCH_API_TOKEN=... (mint one at "
                "https://digbench.ai/account/tokens)"
            )
        self.model_name = model_name
        self.model_version = model_version
        self.notes = notes if (notes is None or isinstance(notes, str)) else json.dumps(notes, ensure_ascii=False)
        self.max_retries = max(1, int(max_retries))
        self.session_timeout = float(session_timeout)
        self.on_retry = on_retry
        self._http = _Http(self.base_url, self._token, timeout=timeout, user_agent=USER_AGENT)

        self.session_id: Optional[str] = None
        self.step_index: int = 0
        self.description: Optional[str] = None
        self.seed: Optional[int] = None
        self.framework_version: Optional[str] = None
        self.move_schema: Optional[dict] = None
        self.last: Optional[Transition] = None

    def _call_once(self, method: str, path: str, payload: Optional[dict] = None,
                   *, timeout: Optional[float] = None) -> dict:
        try:
            status, headers, body = self._http.request(method, path, payload, timeout=timeout)
        except Exception as e:
            raise _Transient(f"network: {type(e).__name__}: {e}") from e
        if 200 <= status < 300:
            return body
        detail = body.get("detail") or body.get("_raw") or body
        if status in (401, 403):
            raise DigBenchAuthError(f"{method} {path} -> HTTP {status}: {detail}")
        if status == 409:
            raise DigBenchProtocolError(f"{method} {path} -> HTTP 409: {detail}")
        if status not in _TRANSIENT_HTTP:
            raise DigBenchError(f"{method} {path} -> HTTP {status}: {detail}")
        ra = headers.get("retry-after") if status == 429 else None
        raise _Transient(f"HTTP {status}: {detail}", retry_after=ra)

    def _call(self, method: str, path: str, payload: Optional[dict] = None,
              *, timeout: Optional[float] = None, before_retry=None) -> dict:
        last = ""
        for attempt in range(self.max_retries):
            try:
                return self._call_once(method, path, payload, timeout=timeout)
            except _Transient as e:
                last, ra = str(e), e.retry_after
            if attempt < self.max_retries - 1:
                if before_retry is not None:
                    adopted = before_retry()
                    if adopted is not None:
                        return adopted
                delay = min(2.0 ** attempt, 30.0)
                if ra:
                    try:
                        delay = max(delay, float(ra))
                    except ValueError:
                        pass
                if self.on_retry:
                    try:
                        self.on_retry(f"bench {method} {path} attempt {attempt + 1}/{self.max_retries} "
                                      f"failed: {last}; retrying in {delay:.0f}s")
                    except Exception:
                        pass
                time.sleep(delay)
        raise DigBenchError(f"{method} {path} failed after {self.max_retries} attempts: {last}")

    def _absorb_session_meta(self, d: dict) -> None:
        self.session_id = str(d.get("session_id") or self.session_id or "")
        self.step_index = _int_or(d.get("step_index"), self.step_index)
        if d.get("description") is not None:
            self.description = str(d["description"])
        if d.get("seed") is not None:
            self.seed = _opt_int(d.get("seed"))
        if d.get("framework_version") is not None:
            self.framework_version = str(d["framework_version"])
        if isinstance(d.get("move_schema"), dict):
            self.move_schema = d["move_schema"]
        if d.get("game"):
            self.game = str(d["game"])

    def start(self) -> Transition:
        if self.session_id:
            raise DigBenchError(
                f"this env already owns session {self.session_id}; a run is bound to ONE session "
                "(use resume(), never start a second game)")
        payload: dict[str, Any] = {"game": self.game}
        if self.model_name:
            payload["model_name"] = self.model_name
        if self.model_version:
            payload["model_version"] = self.model_version
        if self.notes:
            payload["notes"] = self.notes
        started_at = time.time()

        def _adopt() -> Optional[dict]:
            sid = self._find_fresh_session(started_at)
            if not sid:
                return None
            if self.on_retry:
                try:
                    self.on_retry(f"bench POST /sessions: response lost but session {sid} was created "
                                  "server-side — adopting it instead of opening another")
                except Exception:
                    pass
            try:
                return self._call_once("GET", f"/sessions/{sid}")
            except _Transient:
                return None

        d = self._call("POST", "/sessions", payload, timeout=self.session_timeout, before_retry=_adopt)
        self._absorb_session_meta(d)
        self.last = Transition.from_response(d)
        return self.last

    def _find_fresh_session(self, started_at: float, slack_s: float = 120.0) -> Optional[str]:
        try:
            sessions = self._call_once("GET", "/sessions").get("sessions") or []
        except Exception:
            return None
        best: "tuple[float, str] | None" = None
        for s in sessions:
            if s.get("game") != self.game or (s.get("model_name") or None) != (self.model_name or None):
                continue
            if str(s.get("status", "")).lower() not in ("running", "in_progress"):
                continue
            if (s.get("steps") or 0) > 0:
                continue
            created = _parse_ts(s.get("created_at"))
            if created is None or created < started_at - slack_s:
                continue
            sid = str(s.get("session_id") or "")
            if sid and (best is None or created > best[0]):
                best = (created, sid)
        return best[1] if best else None

    def resume(self, session_id: str) -> Transition:
        self.session_id = str(session_id)
        return self.get_session()

    def get_session(self) -> Transition:
        if not self.session_id:
            raise DigBenchError("no session to fetch — call start() or resume(session_id) first")
        d = self._call("GET", f"/sessions/{self.session_id}")
        self._absorb_session_meta(d)
        t = Transition.from_response(d)
        self.last = t
        return t

    def step(self, action: str, *, reasoning: Optional[str] = None) -> Transition:
        if not self.session_id:
            raise DigBenchError("no session — call start() or resume(session_id) first")
        action = str(action)
        idx = self.step_index + 1
        payload: dict[str, Any] = {"step_index": idx, "action": action}
        if reasoning:
            payload["reasoning"] = reasoning[:16_000]
        try:
            d = self._call("POST", f"/sessions/{self.session_id}/step", payload)
        except DigBenchProtocolError:
            try:
                self.get_session()
            except Exception:
                pass
            raise
        t = Transition.from_response(d, action=action)
        expected = self.step_index if t.invalid_action else idx
        if t.step_index != expected:
            self.step_index = t.step_index
            self.last = t
            raise DigBenchProtocolError(
                f"server step_index {t.step_index} != expected {expected} after action {action!r}")
        self.step_index = t.step_index
        self.last = t
        return t

    def close(self) -> None:
        self._http.close()

    def __repr__(self) -> str:
        return f"DigEnv(game={self.game!r} session={self.session_id!r} step_index={self.step_index})"
