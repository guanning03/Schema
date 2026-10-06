from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import httpx

ToolHandler = Callable[[str, dict], "tuple[str, bool, bool]"]
EventHandler = Callable[[dict], None]

_BASE_URL = "https://chatgpt.com/backend-api/codex/responses"
_TOKEN_URL = "https://auth.openai.com/oauth/token"
_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
_ORIGINATOR = "codex_cli_rs"
_CLIENT_VERSION = "0.153.4"
_AUTH_PATH = Path(os.path.expanduser("~/.codex/auth.json"))
_MAX_TOOL_ITERATIONS = 100

_POOL_ENV = "ARC_CODEX_POOL"
_POOL_GLOB = ".codex-arc-agent*"
_FALLBACK_LIMIT_COOLDOWN = 300.0
_COOLDOWN_CAP = 1800.0
_LIMIT_MSG_RE = re.compile(r"usage.?limit|rate.?limit|quota|too many requests", re.IGNORECASE)

_CONTEXT_WINDOW = 272_000
_COMPACT_PCT = 90
_TAIL_PCT = 7
_APPROX_BYTES_PER_TOKEN = 4
_MAX_COMPACT_FAILURES = 3

COMPACT_PROMPT = """\
CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.

You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary for
another instance of yourself that will resume this exact task with no other memory
of the earlier conversation.

Rules:
- Do NOT call any tool. Output plain text only.
- Treat this as a handoff to your future self, not a user-facing status update.
- Be concise, structured, and focused on what is needed to continue the work.
- Your final answer must contain exactly two plain-text blocks:
  1. <analysis>...</analysis>
  2. <summary>...</summary>

In <analysis>, identify the durable context worth carrying forward:
- What the task is and the current overall progress / key decisions made.
- The current world-model hypotheses and which mechanics are confirmed vs. guessed.
- Important facts about the game, the current level, and the plan.
- Files written in the workdir (world_model.py, notes.md, etc.) and what they contain.
- What remains to be done and the immediate next step.

In <summary>, write only the compact handoff your future self will read as
authoritative background. Include the primary task, current progress and decisions,
the world model's state, key confirmed facts, open questions, and next steps.
"""

SUMMARY_WRAP = (
    "[Earlier conversation has been compacted to save context. The text below is a "
    "handoff summary of everything you did and learned before this point — treat it "
    "as authoritative background for your own prior work:]\n\n{summary}"
)


@dataclass
class TurnResult:
    final_text: str = ""
    usage: dict = field(default_factory=dict)
    usage_total: dict = field(default_factory=dict)
    committed: bool = False
    response_id: Optional[str] = None


def _usage_add(total: dict, usage: dict) -> None:
    if not isinstance(usage, dict):
        return
    total["requests"] = total.get("requests", 0) + 1
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        v = usage.get(key)
        if isinstance(v, (int, float)):
            total[key] = total.get(key, 0) + int(v)
    cached = (usage.get("input_tokens_details") or {}).get("cached_tokens")
    if isinstance(cached, (int, float)):
        total["cached_tokens"] = total.get("cached_tokens", 0) + int(cached)
    reasoning = (usage.get("output_tokens_details") or {}).get("reasoning_tokens")
    if isinstance(reasoning, (int, float)):
        total["reasoning_tokens"] = total.get("reasoning_tokens", 0) + int(reasoning)


class CodexAuthError(RuntimeError):
    pass


class CodexLimitError(RuntimeError):
    def __init__(self, kind: str, reset_at: Optional[float], detail: str = "") -> None:
        super().__init__(detail or kind)
        self.kind = kind
        self.reset_at = reset_at


def _find_reset_seconds(obj) -> Optional[float]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("resets_in_seconds", "reset_after_seconds", "retry_after") and isinstance(v, (int, float)):
                return float(v)
            got = _find_reset_seconds(v)
            if got is not None:
                return got
    elif isinstance(obj, list):
        for v in obj:
            got = _find_reset_seconds(v)
            if got is not None:
                return got
    return None


def _classify_limit(status: int, headers, body_text: str) -> "Optional[tuple[str, Optional[float]]]":
    if status != 429:
        return None
    secs: Optional[float] = None
    try:
        secs = _find_reset_seconds(json.loads(body_text))
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


class CodexDriver:

    def __init__(self, *, model: str, reasoning: Optional[str] = None, timeout: float = 600.0) -> None:
        self.model = model
        self.reasoning = reasoning
        self.timeout = timeout
        self._turns: list[list[dict]] = []
        self._session_uuid = uuid.uuid4().hex
        self._last_input_tokens = 0
        self._compact_failures = 0
        self._notify = lambda m: print(m, file=sys.stderr, flush=True)
        self.context_window = _CONTEXT_WINDOW
        self._compact_threshold = int(self.context_window * _COMPACT_PCT / 100)
        self._tail_fraction = _TAIL_PCT / 100
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

    def _rotate_account(self, kind: str, reset_at: Optional[float]) -> None:
        assert self.pool is not None
        self.pool.mark_limited(str(self._auth_path.parent), kind, reset_at)
        self._auth_path = Path(self.pool.acquire()) / "auth.json"
        self._auth = self._load_auth()

    def _refresh(self) -> bool:
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

    def _headers(self) -> dict:
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

    def _flatten(self, upto: Optional[int] = None) -> list[dict]:
        turns = self._turns if upto is None else self._turns[:upto]
        return [item for turn in turns for item in turn]

    def _commit_turn(self, cur: list[dict], usage: dict) -> None:
        self._turns.append(cur)
        self._last_input_tokens = self._input_tokens_from_usage(usage)

    def _note_input_tokens(self, usage: dict) -> None:
        tokens = self._input_tokens_from_usage(usage)
        if tokens > 0:
            self._last_input_tokens = tokens

    @staticmethod
    def _input_tokens_from_usage(usage: dict) -> int:
        if not isinstance(usage, dict):
            return 0
        for key in ("input_tokens", "prompt_tokens"):
            v = usage.get(key)
            if isinstance(v, (int, float)):
                return int(v)
        return 0

    @staticmethod
    def _estimate_item_tokens(item: dict) -> int:
        total = 0
        content = item.get("content")
        if isinstance(content, list):
            for blk in content:
                if isinstance(blk, dict):
                    total += len(blk.get("text", "")) // _APPROX_BYTES_PER_TOKEN
        elif isinstance(content, str):
            total += len(content) // _APPROX_BYTES_PER_TOKEN
        for key in ("arguments", "output", "name"):
            v = item.get(key)
            if isinstance(v, str):
                total += len(v) // _APPROX_BYTES_PER_TOKEN
        return total + 8

    def _estimate_tokens(self, items: list[dict]) -> int:
        return sum(self._estimate_item_tokens(it) for it in items)

    @staticmethod
    def _extract_summary(text: str) -> Optional[str]:
        if not text or not text.strip():
            return None
        m = re.search(r"<summary>(.*?)</summary>", text, re.S | re.I)
        if m:
            return m.group(1).strip() or None
        cleaned = re.sub(r"<analysis>.*?</analysis>", "", text, flags=re.S | re.I).strip()
        return cleaned or text.strip()

    def _maybe_compact(self, system_prompt: str) -> bool:
        if self._compact_failures >= _MAX_COMPACT_FAILURES:
            return False
        if len(self._turns) < 2 or self._last_input_tokens < self._compact_threshold:
            return False

        tail_budget = int(self.context_window * self._tail_fraction)
        acc, keep = 0, 0
        for turn in reversed(self._turns):
            tk = self._estimate_tokens(turn)
            if keep >= 1 and acc + tk > tail_budget:
                break
            acc += tk
            keep += 1
        k = len(self._turns) - keep
        if k <= 0:
            return False

        prefix = self._flatten(upto=k)
        self._notify(
            f"[codex] context ~{self._last_input_tokens} tok ≥ "
            f"{self._compact_threshold} → compacting {k} old turn(s), keeping {keep}…"
        )
        summary = self._run_compaction(system_prompt, prefix)
        if summary is None:
            self._compact_failures += 1
            self._notify(f"[codex] compaction failed ({self._compact_failures}/{_MAX_COMPACT_FAILURES}).")
            return False
        self._compact_failures = 0
        self._turns = [[self._summary_item(summary)]] + self._turns[k:]
        self._last_input_tokens = self._estimate_tokens(self._flatten())
        self._notify(f"[codex] compacted → ~{self._last_input_tokens} tok in history.")
        return True

    def export_sessions(self, dest: Path) -> None:
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "session_uuid": self._session_uuid,
            "last_input_tokens": self._last_input_tokens,
            "turns": self._turns,
        }
        (dest / "client_session.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    def import_sessions(self, src: Path) -> None:
        path = Path(src) / "client_session.json"
        if not path.is_file():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        turns = data.get("turns")
        if not isinstance(turns, list):
            return
        self._turns = [t for t in turns if isinstance(t, list)]
        if isinstance(data.get("session_uuid"), str) and data["session_uuid"]:
            self._session_uuid = data["session_uuid"]
        lit = data.get("last_input_tokens")
        self._last_input_tokens = (
            int(lit) if isinstance(lit, (int, float)) else self._estimate_tokens(self._flatten())
        )

    @staticmethod
    def _to_responses_tools(tools: list[dict]) -> list[dict]:
        return [
            {
                "type": "function",
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t["input_schema"],
            }
            for t in tools
        ]

    @staticmethod
    def _user_item(user_message: dict) -> dict:
        content = user_message.get("content", "")
        if isinstance(content, str):
            return {"role": "user", "content": [{"type": "input_text", "text": content}]}
        return {"role": "user", "content": [
            {"type": "input_text", "text": blk.get("text", "")}
            for blk in content if isinstance(blk, dict) and blk.get("type") == "text"]}

    def _build_body(self, system_prompt: str, input_items: list[dict], tools: list[dict]) -> dict:
        body: dict = {"model": self.model, "input": input_items, "stream": True, "store": False}
        body["prompt_cache_key"] = self._session_uuid
        if system_prompt:
            body["instructions"] = system_prompt
        if self.reasoning:
            body["reasoning"] = {"effort": self.reasoning, "summary": "auto"}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        return body

    def _summary_item(self, summary_text: str) -> dict:
        return {"role": "user",
                "content": [{"type": "input_text", "text": SUMMARY_WRAP.format(summary=summary_text)}]}

    def _run_compaction(self, system_prompt: str, prefix_items: list[dict]) -> Optional[str]:
        compact_user = {"role": "user", "content": [{"type": "input_text", "text": COMPACT_PROMPT}]}
        body: dict = {
            "model": self.model,
            "input": prefix_items + [compact_user],
            "stream": True,
            "store": False,
            "prompt_cache_key": self._session_uuid,
        }
        if system_prompt:
            body["instructions"] = system_prompt
        try:
            text, _tool_calls, _rid, _usage = self._stream_once(body, None)
        except Exception as e:
            self._notify(f"[codex] compaction request error: {type(e).__name__}: {e}")
            return None
        return self._extract_summary(text)

    def run_turn(
        self,
        system_prompt: str,
        build_user_message: "Callable[[bool], dict]",
        tools: list[dict],
        on_tool_call: ToolHandler,
        on_event: Optional[EventHandler] = None,
    ) -> TurnResult:
        if not self.available():
            raise CodexAuthError("no codex access_token; run `codex login` first.")

        continuing = bool(self._turns)
        if continuing:
            self._maybe_compact(system_prompt)

        user_message = build_user_message(continuing)
        fn_tools = self._to_responses_tools(tools)
        result = TurnResult()

        base = self._flatten()
        cur: list[dict] = [self._user_item(user_message)]

        committed = False
        for _ in range(_MAX_TOOL_ITERATIONS):
            body = self._build_body(system_prompt, base + cur, fn_tools)
            text, tool_calls, response_id, usage = self._stream_once(body, on_event)
            result.final_text = text
            result.response_id = response_id
            result.usage = usage
            _usage_add(result.usage_total, usage)

            if text.strip():
                cur.append({"role": "assistant", "content": [{"type": "output_text", "text": text}]})

            if not tool_calls:
                break

            stop = False
            for tc in tool_calls:
                cur.append({
                    "type": "function_call",
                    "call_id": tc["call_id"],
                    "name": tc["name"],
                    "arguments": tc["arguments"],
                })
                try:
                    args = json.loads(tc["arguments"] or "{}")
                    if not isinstance(args, dict):
                        args = {}
                except (ValueError, TypeError):
                    args = {}
                output, _is_error, want_stop = on_tool_call(tc["name"], args)
                cur.append({
                    "type": "function_call_output",
                    "call_id": tc["call_id"],
                    "output": output,
                })
                if want_stop:
                    stop = True
            if stop:
                committed = True
                break
            self._note_input_tokens(usage)
            if self._maybe_compact(system_prompt):
                base = self._flatten()
        self._commit_turn(cur, result.usage)
        result.committed = committed
        return result

    def _stream_once(
        self, body: dict, on_event: Optional[EventHandler]
    ) -> "tuple[str, list[dict], Optional[str], dict]":
        refreshed = False
        while True:
            try:
                return self._do_stream(body, on_event)
            except CodexLimitError as e:
                if self.pool is None:
                    raise CodexAuthError(f"codex usage limit: {e}") from e
                self._rotate_account(e.kind, e.reset_at)
                refreshed = False
                continue
            except httpx.HTTPStatusError as e:
                if e.response is not None and e.response.status_code == 401 and not refreshed:
                    refreshed = True
                    if self._refresh():
                        continue
                    if self.pool is not None and len(self.pool) > 1:
                        self._rotate_account("auth-failed (refresh failed)", None)
                        refreshed = False
                        continue
                detail = ""
                try:
                    detail = e.response.text[:500] if e.response is not None else ""
                except Exception:
                    pass
                raise CodexAuthError(f"codex HTTP error: {e}; {detail}") from e

    def _do_stream(
        self, body: dict, on_event: Optional[EventHandler]
    ) -> "tuple[str, list[dict], Optional[str], dict]":
        text = ""
        response_id: Optional[str] = None
        usage: dict = {}
        partial: dict[str, dict] = {}

        with httpx.Client(timeout=self.timeout) as client:
            with client.stream("POST", _BASE_URL, json=body, headers=self._headers()) as resp:
                if resp.status_code >= 400:
                    body_text = resp.read().decode("utf-8", "replace")
                    limit = _classify_limit(resp.status_code, resp.headers, body_text)
                    if limit is not None:
                        raise CodexLimitError(limit[0], limit[1], body_text[:300])
                    raise httpx.HTTPStatusError(
                        f"{resp.status_code}: {body_text[:800]}", request=resp.request, response=resp
                    )
                for line in resp.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        ev = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    etype = ev.get("type", "")

                    if etype == "response.output_text.delta":
                        text += ev.get("delta", "") or ""
                    elif etype == "response.output_item.added":
                        item = ev.get("item") or {}
                        if item.get("type") == "function_call":
                            partial[item.get("id", "")] = {
                                "call_id": item.get("call_id", ""),
                                "name": item.get("name", ""),
                                "arguments": item.get("arguments", "") or "",
                            }
                    elif etype == "response.function_call_arguments.delta":
                        iid = ev.get("item_id", "")
                        if iid in partial:
                            partial[iid]["arguments"] += ev.get("delta", "") or ""
                    elif etype == "response.function_call_arguments.done":
                        iid = ev.get("item_id", "")
                        if iid in partial and ev.get("arguments") is not None:
                            partial[iid]["arguments"] = ev["arguments"]
                    elif etype == "response.output_item.done":
                        item = ev.get("item") or {}
                        if item.get("type") == "function_call":
                            iid = item.get("id", "")
                            if iid in partial:
                                if item.get("call_id"):
                                    partial[iid]["call_id"] = item["call_id"]
                                if item.get("name"):
                                    partial[iid]["name"] = item["name"]
                                if item.get("arguments") is not None:
                                    partial[iid]["arguments"] = item["arguments"]
                    elif etype in ("response.completed", "response.incomplete"):
                        r = ev.get("response") or {}
                        response_id = r.get("id")
                        usage = r.get("usage") or {}
                    elif etype == "error":
                        msg = str(ev.get("message") or ev)
                        if _LIMIT_MSG_RE.search(msg):
                            secs = _find_reset_seconds(ev)
                            raise CodexLimitError(
                                "limit", time.time() + secs if secs is not None else None, msg[:300]
                            )
                        raise CodexAuthError(f"codex stream error: {msg}")

                    if on_event and etype.startswith("response."):
                        on_event(ev)

        tool_calls = [
            {"call_id": p["call_id"], "name": p["name"], "arguments": p["arguments"]}
            for p in partial.values()
            if p.get("call_id") and p.get("name")
        ]
        return text, tool_calls, response_id, usage
