from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

ToolHandler = Callable[[str, dict], "tuple[str, bool, bool]"]
EventHandler = Callable[[dict], None]
MessageBuilder = Callable[[bool], dict]

_SERVER_NAME = "locus"

_DISALLOWED_BUILTINS = [
    "Bash", "BashOutput", "KillShell", "Edit", "Write", "Read", "NotebookEdit",
    "Glob", "Grep", "WebFetch", "WebSearch", "Task", "TodoWrite", "SlashCommand",
]


@dataclass
class TurnResult:
    final_text: str = ""
    usage: dict = field(default_factory=dict)
    claude_session_id: Optional[str] = None
    stderr: str = ""
    committed: bool = False
    error_kind: Optional[str] = None
    limit_reset_at: Optional[float] = None


_LIMIT_RE = re.compile(
    r"usage limit reached|5\s*-?\s*hour limit|weekly limit|session limit|rate.?limit"
    r"|too many requests|\b429\b|quota|out of usage credits",
    re.I,
)
_OVERLOAD_RE = re.compile(r"overloaded|\b529\b|\b503\b|service unavailable", re.I)
_INFRA_RE = re.compile(
    r"OCI runtime|\bcrun\b|\brunc\b|create keyring|disk quota exceeded"
    r"|no space left on device|cannot allocate memory",
    re.I,
)

_BILLING_DIVERTING_ENV = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
)
_RESET_TS_RE = re.compile(r"limit reached\s*\|\s*(\d{10})", re.I)

_FALLBACK_LIMIT_COOLDOWN = 300.0
_FALLBACK_OVERLOAD_COOLDOWN = 30.0
_INFRA_COOLDOWN = 60.0
_INFRA_MAX_RETRIES = 10
_TURN_IDLE_TIMEOUT = 240.0


def _report_infra(result: "TurnResult", attempt: int, notify: "Callable[[str], None]") -> None:
    blob = (result.stderr or result.final_text or "").strip().replace("\n", " ")
    m = _INFRA_RE.search(blob)
    excerpt = blob[max(0, m.start() - 120):m.end() + 120] if m else blob[-240:]
    notify(f"[claude] infrastructure failure (not an account limit), retrying this turn in "
           f"{_INFRA_COOLDOWN:.0f}s ({attempt}/{_INFRA_MAX_RETRIES}): {excerpt}")


def _classify_failure(blob: str, is_error: bool) -> "tuple[Optional[str], Optional[float]]":
    if _INFRA_RE.search(blob):
        return "infra", None
    hit_limit = bool(_LIMIT_RE.search(blob))
    hit_overload = bool(_OVERLOAD_RE.search(blob))
    if hit_limit:
        m = _RESET_TS_RE.search(blob)
        return "limit", (float(m.group(1)) if m else None)
    if hit_overload and is_error:
        return "overloaded", None
    return None, None


_USAGE_ENDPOINT = "https://api.anthropic.com/api/oauth/usage"
_OAUTH_BETA = "oauth-2025-04-20"
_PROBE_TTL = 15.0
_SWITCH_AT = 95.0
_WEEKLY_SWITCH_AT = 98.0


def _keychain_service(config_dir: str) -> str:
    abspath = os.path.abspath(os.path.expanduser(config_dir))
    if abspath == os.path.abspath(os.path.expanduser("~/.claude")):
        return "Claude Code-credentials"
    return "Claude Code-credentials-" + hashlib.sha256(abspath.encode()).hexdigest()[:8]


def _read_oauth_token(config_dir: str) -> Optional[str]:
    blob: Optional[str] = None
    cred = os.path.join(os.path.expanduser(config_dir), ".credentials.json")
    if os.path.isfile(cred):
        try:
            blob = Path(cred).read_text()
        except Exception:
            blob = None
    if blob is None and sys.platform == "darwin":
        try:
            blob = subprocess.run(
                ["security", "find-generic-password", "-s", _keychain_service(config_dir), "-w"],
                capture_output=True, text=True, timeout=10,
            ).stdout
        except Exception:
            blob = None
    if not blob:
        return None
    try:
        return (json.loads(blob).get("claudeAiOauth") or {}).get("accessToken")
    except Exception:
        return None


def _parse_iso_ts(ts: str) -> Optional[float]:
    try:
        return datetime.datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def _is_weekly_window(key: str) -> bool:
    return key.startswith("seven_day")


def _probe_account_usage(
    config_dir: str,
) -> "Optional[tuple[list[tuple[str, float, Optional[float]]], bool]]":
    token = _read_oauth_token(config_dir)
    if not token:
        return None
    req = urllib.request.Request(
        _USAGE_ENDPOINT,
        headers={
            "Authorization": f"Bearer {token}",
            "anthropic-beta": _OAUTH_BETA,
            "User-Agent": "arc-world-model-account-pool",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode())
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    eu = data.get("extra_usage")
    extra_enabled = bool(eu.get("is_enabled")) if isinstance(eu, dict) else False
    windows: "list[tuple[str, float, Optional[float]]]" = []
    for key, win in data.items():
        if key in ("extra_usage", "spend") or not isinstance(win, dict):
            continue
        u = win.get("utilization")
        if isinstance(u, (int, float)):
            windows.append((key, float(u), _parse_iso_ts(win.get("resets_at") or "")))
    limits = data.get("limits")
    if isinstance(limits, list):
        for lim in limits:
            if not isinstance(lim, dict) or lim.get("kind") != "weekly_scoped":
                continue
            pct = lim.get("percent")
            if not isinstance(pct, (int, float)):
                continue
            model = ((lim.get("scope") or {}).get("model") or {}).get("display_name") or "scoped"
            windows.append((
                f"seven_day_scoped_{str(model).lower()}",
                float(pct),
                _parse_iso_ts(lim.get("resets_at") or ""),
            ))
    if not windows:
        return None
    return windows, extra_enabled


class AccountPool:

    def __init__(self, dirs: list[str], notify: Optional[Callable[[str], None]] = None) -> None:
        self._dirs = list(dirs)
        self._cooldown: dict[str, float] = {}
        self._idx = 0
        self._lock = threading.Lock()
        self._notify = notify or (lambda m: print(m, file=sys.stderr, flush=True))
        self._probe_cache: dict = {}
        self._protected: set[str] = set()

    def __len__(self) -> int:
        return len(self._dirs)

    @staticmethod
    def _threshold_for(window_key: str) -> float:
        return _WEEKLY_SWITCH_AT if _is_weekly_window(window_key) else _SWITCH_AT

    def _is_protected(self, config_dir: str) -> bool:
        return os.path.abspath(os.path.expanduser(config_dir)) in self._protected

    def _probe_cached(self, config_dir: str):
        now = time.time()
        hit = self._probe_cache.get(config_dir)
        if hit and now - hit[0] < _PROBE_TTL and not self._is_protected(config_dir):
            return hit[1]
        res = _probe_account_usage(config_dir)
        self._probe_cache[config_dir] = (now, res)
        if res is not None and res[1]:
            self._protected.add(os.path.abspath(os.path.expanduser(config_dir)))
        return res

    @classmethod
    def discover(cls, notify: Optional[Callable[[str], None]] = None) -> Optional["AccountPool"]:
        raw = os.environ.get("ARC_CLAUDE_POOL", "").strip()
        if raw:
            dirs = [os.path.expanduser(p) for p in re.split(r"[:,]", raw) if p.strip()]
        else:
            dirs = sorted(str(p) for p in Path.home().glob(".claude-arc-agent*") if p.is_dir())
        dirs = [d for d in dirs if os.path.isdir(d)]
        if len(dirs) < 2:
            return None
        pool = cls(dirs, notify=notify)
        pool._seed_protection()
        return pool

    def _seed_protection(self) -> None:
        for d in self._dirs:
            try:
                self._probe_cached(d)
            except Exception:
                pass
        prot = sorted(p.split("/")[-1] for p in self._protected)
        if prot:
            self._notify(
                f"[account-pool] protected accounts (extra_usage enabled): {', '.join(prot)} — "
                f"never used while their usage cannot be probed."
            )

    def acquire(self) -> str:
        while True:
            now = time.time()
            with self._lock:
                for d in list(self._cooldown):
                    if self._cooldown[d] <= now:
                        del self._cooldown[d]
                n = len(self._dirs)
                candidates = [
                    self._dirs[(self._idx + off) % n]
                    for off in range(n)
                    if self._dirs[(self._idx + off) % n] not in self._cooldown
                ]
            for d in candidates:
                pr = self._probe_cached(d)
                if pr is None:
                    if self._is_protected(d):
                        self._notify(
                            f"[account-pool] {d} is protected but its usage cannot be probed — "
                            f"skipping, re-probe in {int(_FALLBACK_OVERLOAD_COOLDOWN)}s."
                        )
                        self.mark_limited(
                            d, "probe failed (protected, fail-closed)",
                            time.time() + _FALLBACK_OVERLOAD_COOLDOWN,
                        )
                        continue
                    with self._lock:
                        self._idx = self._dirs.index(d)
                    return d
                windows, _ = pr
                breached = [
                    (key, util, reset)
                    for (key, util, reset) in windows
                    if util >= self._threshold_for(key)
                ]
                if not breached:
                    with self._lock:
                        self._idx = self._dirs.index(d)
                    return d
                resets = [r for (_, _, r) in breached if r is not None]
                cool_to = max(resets) if resets else None
                label = ", ".join(
                    f"{key} {util:.0f}%≥{self._threshold_for(key):.0f}%"
                    for (key, util, _) in breached
                )
                self._probe_cache.pop(d, None)
                self.mark_limited(d, label, cool_to)

            with self._lock:
                if not self._cooldown:
                    continue
                wake = min(self._cooldown.values())
            sleep_for = max(1.0, wake - time.time())
            self._notify(
                f"[account-pool] all {len(self._dirs)} accounts are at their limit / cooling down; "
                f"sleeping {int(sleep_for)}s until the earliest reset "
                f"({time.strftime('%H:%M:%S', time.localtime(wake))})…"
            )
            time.sleep(min(sleep_for, 60.0))

    def mark_limited(self, config_dir: str, kind: str, reset_at: Optional[float]) -> None:
        if reset_at is None:
            cd = _FALLBACK_OVERLOAD_COOLDOWN if kind == "overloaded" else _FALLBACK_LIMIT_COOLDOWN
            reset_at = time.time() + cd
        with self._lock:
            self._cooldown[config_dir] = reset_at
            self._idx = (self._idx + 1) % len(self._dirs)
        when = time.strftime("%H:%M:%S", time.localtime(reset_at))
        self._notify(
            f"[account-pool] {config_dir} hit {kind} (expected reset {when}), switching account."
        )


class ClaudeDriver:
    def __init__(
        self,
        *,
        model: Optional[str] = None,
        effort: Optional[str] = None,
        cwd: Optional[str] = None,
        container: Optional["object"] = None,
    ) -> None:
        self.model = model
        self.effort = effort
        self.cwd = cwd or os.getcwd()
        self._container = container
        self._exe = shutil.which("claude")
        self.pool = AccountPool.discover()
        self._session_id: Optional[str] = None

    def available(self) -> bool:
        return self._exe is not None

    def _build_args(self, system_prompt: str, resume_session_id: Optional[str] = None) -> list[str]:
        args = [
            self._exe,
            "--print",
            "--output-format", "stream-json",
            "--verbose",
            "--include-partial-messages",
            "--input-format", "stream-json",
            "--system-prompt", system_prompt,
            "--exclude-dynamic-system-prompt-sections",
            "--permission-mode", "bypassPermissions",
        ]
        if resume_session_id:
            args += ["--resume", resume_session_id]
        if self.model:
            args += ["--model", self.model]
        if self.effort:
            args += ["--effort", self.effort]
        args += ["--disallowed-tools", *_DISALLOWED_BUILTINS]
        return args

    @staticmethod
    def _to_mcp_tools(tools: list[dict]) -> list[dict]:
        return [
            {"name": t["name"], "description": t.get("description", ""), "inputSchema": t["input_schema"]}
            for t in tools
        ]

    def run_turn(
        self,
        system_prompt: str,
        build_user_message: MessageBuilder,
        tools: list[dict],
        on_tool_call: ToolHandler,
        on_event: Optional[EventHandler] = None,
    ) -> TurnResult:
        if self._exe is None:
            raise RuntimeError(
                "`claude` CLI not found on PATH. Install @anthropic-ai/claude-code and log in."
            )

        tools = self._to_mcp_tools(tools)

        if self.pool is None:
            stall_retries = 0
            infra_retries = 0
            while True:
                result = self._run_once(
                    system_prompt, build_user_message, tools, on_tool_call, on_event, None
                )
                if result.error_kind == "infra" and infra_retries < _INFRA_MAX_RETRIES:
                    infra_retries += 1
                    _report_infra(result, infra_retries, print)
                    time.sleep(_INFRA_COOLDOWN)
                    continue
                if result.error_kind in ("limit", "overloaded"):
                    fallback = (_FALLBACK_OVERLOAD_COOLDOWN if result.error_kind == "overloaded"
                                else _FALLBACK_LIMIT_COOLDOWN)
                    wake = result.limit_reset_at or (time.time() + fallback)
                    sleep_for = max(1.0, wake - time.time())
                    print(f"[claude] hit {result.error_kind} "
                          f"(expected reset {time.strftime('%H:%M:%S', time.localtime(wake))}), "
                          f"sleeping {int(sleep_for)}s before retrying this turn…", flush=True)
                    while time.time() < wake:
                        time.sleep(min(60.0, max(1.0, wake - time.time())))
                    continue
                if result.error_kind == "stalled" and stall_retries < 3:
                    stall_retries += 1
                    print(f"[turn] claude produced no output and was killed; retrying with a fresh "
                          f"connection ({stall_retries}/3)", flush=True)
                    continue
                return result

        stall_retries = 0
        infra_retries = 0
        while True:
            config_dir = self.pool.acquire()
            result = self._run_once(
                system_prompt, build_user_message, tools, on_tool_call, on_event, config_dir
            )
            if result.error_kind == "infra" and infra_retries < _INFRA_MAX_RETRIES:
                infra_retries += 1
                _report_infra(result, infra_retries, self.pool._notify)
                time.sleep(_INFRA_COOLDOWN)
                continue
            if result.error_kind in ("limit", "overloaded"):
                self.pool.mark_limited(config_dir, result.error_kind, result.limit_reset_at)
                continue
            if result.error_kind == "stalled" and stall_retries < 3:
                stall_retries += 1
                self.pool._notify(
                    f"[turn] claude produced no output and was killed; retrying with a fresh "
                    f"connection ({stall_retries}/3)"
                )
                continue
            return result

    def _run_once(
        self,
        system_prompt: str,
        build_user_message: MessageBuilder,
        tools: list[dict],
        on_tool_call: ToolHandler,
        on_event: Optional[EventHandler],
        config_dir: Optional[str],
    ) -> TurnResult:
        claude_env: dict[str, str] = {"CLAUDE_CODE_ENTRYPOINT": "sdk-py"}
        if config_dir:
            claude_env["CLAUDE_CONFIG_DIR"] = config_dir

        resume_session_id = self._session_id
        if resume_session_id:
            self._install_session(config_dir, resume_session_id)
        continuing = resume_session_id is not None
        user_message = build_user_message(continuing)

        base_argv = self._build_args(system_prompt, resume_session_id)
        if self._container is not None:
            argv = self._container.wrap(base_argv, config_dir, claude_env)
            popen_env = self._container.popen_env
            popen_cwd = None
        else:
            popen_env = dict(os.environ, **claude_env)
            for k in _BILLING_DIVERTING_ENV:
                popen_env.pop(k, None)
            argv = base_argv
            popen_cwd = self.cwd
        proc = subprocess.Popen(
            argv,
            cwd=popen_cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=popen_env,
        )

        err_buf: list[str] = []

        def _drain_err() -> None:
            try:
                for line in proc.stderr:
                    err_buf.append(line)
            except Exception:
                pass

        threading.Thread(target=_drain_err, daemon=True).start()

        _last_act = [time.time()]
        _stalled = [False]
        _in_tool = [False]

        def _watchdog() -> None:
            while proc.poll() is None:
                if not _in_tool[0] and time.time() - _last_act[0] > _TURN_IDLE_TIMEOUT:
                    _stalled[0] = True
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    return
                time.sleep(3.0)

        threading.Thread(target=_watchdog, daemon=True).start()

        result = TurnResult()
        init_id = f"req_init_{uuid.uuid4()}"
        self._write(proc, {
            "type": "control_request",
            "request_id": init_id,
            "request": {"subtype": "initialize", "sdkMcpServers": [_SERVER_NAME]},
        })

        init_done = False
        prompt_sent = False
        got_result = False
        try:
            for raw in proc.stdout:
                _last_act[0] = time.time()
                line = raw.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue

                sid = msg.get("session_id")
                if isinstance(sid, str) and sid:
                    result.claude_session_id = sid

                mtype = msg.get("type")
                if mtype == "control_response":
                    resp = msg.get("response") or {}
                    if resp.get("request_id") == init_id and resp.get("subtype") == "success":
                        init_done = True
                elif mtype == "control_request":
                    _in_tool[0] = True
                    try:
                        stop = self._handle_control_request(proc, msg, tools, on_tool_call)
                    finally:
                        _last_act[0] = time.time()
                        _in_tool[0] = False
                    if stop:
                        result.committed = True
                elif mtype in ("stream_event", "assistant"):
                    if on_event:
                        on_event(msg)
                elif mtype == "result":
                    got_result = True
                    result.final_text = msg.get("result", "") or ""
                    result.usage = msg.get("usage", {}) or {}
                    blob = f"{result.final_text}\n{''.join(err_buf)}"
                    result.error_kind, result.limit_reset_at = _classify_failure(
                        blob, bool(msg.get("is_error"))
                    )
                    break

                if init_done and not prompt_sent:
                    self._write(proc, {
                        "type": "user",
                        "session_id": "",
                        "message": user_message,
                        "parent_tool_use_id": None,
                    })
                    prompt_sent = True
        finally:
            self._shutdown(proc)
            result.stderr = "".join(err_buf)
            if not got_result and not result.committed:
                rc = proc.returncode
                result.error_kind, result.limit_reset_at = _classify_failure(
                    result.stderr, is_error=(rc not in (0, None)),
                )
                if _stalled[0]:
                    result.error_kind, result.limit_reset_at = "stalled", None
        if result.error_kind in ("limit", "overloaded"):
            pass
        elif result.claude_session_id and (got_result or result.committed):
            self._session_id = result.claude_session_id
            self._capture_session(config_dir, result.claude_session_id)
        elif continuing:
            self._session_id = None
        return result

    def reset_session(self) -> None:
        self._session_id = None

    @staticmethod
    def _project_slug(cwd: str) -> str:
        return re.sub(r"[^a-zA-Z0-9]", "-", os.path.abspath(os.path.expanduser(cwd)))

    def _transcript_path(self, config_dir: Optional[str], sid: str, cwd: str) -> Path:
        base = config_dir or os.environ.get("CLAUDE_CONFIG_DIR") or "~/.claude"
        return Path(base).expanduser() / "projects" / self._project_slug(cwd) / f"{sid}.jsonl"

    def _canonical_path(self, sid: str) -> Path:
        return Path(self.cwd) / "session_live" / f"{sid}.jsonl"

    def _install_session(self, config_dir: Optional[str], sid: str) -> None:
        canon = self._canonical_path(sid)
        if not canon.is_file():
            return
        dst = self._transcript_path(config_dir, sid, self.cwd)
        try:
            if dst.is_file() and dst.stat().st_size == canon.stat().st_size \
                    and dst.stat().st_mtime >= canon.stat().st_mtime:
                return
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(canon, dst)
        except OSError:
            pass

    def _capture_session(self, config_dir: Optional[str], sid: str) -> None:
        src = self._transcript_path(config_dir, sid, self.cwd)
        if not src.is_file():
            return
        canon = self._canonical_path(sid)
        try:
            canon.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, canon)
        except OSError:
            return
        try:
            for old in canon.parent.glob("*.jsonl"):
                if old.name != f"{sid}.jsonl":
                    old.unlink()
        except OSError:
            pass

    def export_sessions(self, dest: Path) -> None:
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        for old in dest.glob("*.jsonl"):
            try:
                old.unlink()
            except OSError:
                pass
        sid = self._session_id
        if sid:
            canon = self._canonical_path(sid)
            if canon.is_file():
                try:
                    shutil.copy2(canon, dest / f"{sid}.jsonl")
                except OSError:
                    sid = None
            else:
                sid = None
        (dest / "sessions.json").write_text(
            json.dumps({"cwd": self.cwd, "sid": sid}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def import_sessions(self, src: Path) -> None:
        src = Path(src)
        meta_path = src / "sessions.json"
        if not meta_path.is_file():
            self._session_id = None
            return
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self._session_id = None
            return
        sid = meta.get("sid")
        if not sid:
            self._session_id = None
            return
        dump = src / f"{sid}.jsonl"
        if not dump.is_file():
            self._session_id = None
            return
        canon = self._canonical_path(sid)
        try:
            canon.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dump, canon)
        except OSError:
            self._session_id = None
            return
        self._session_id = sid

    def _handle_control_request(
        self, proc: subprocess.Popen, message: dict, tools: list[dict], on_tool_call: ToolHandler
    ) -> bool:
        rid = message.get("request_id", "")
        request = message.get("request") or {}
        subtype = request.get("subtype", "")
        stop = False

        if subtype == "mcp_message":
            inner = request.get("message") or {}
            mcp_resp, stop = self._handle_mcp(inner, tools, on_tool_call)
            resp = {
                "type": "control_response",
                "response": {
                    "subtype": "success",
                    "request_id": rid,
                    "response": {"mcp_response": mcp_resp},
                },
            }
        elif subtype == "can_use_tool":
            resp = {
                "type": "control_response",
                "response": {
                    "subtype": "success",
                    "request_id": rid,
                    "response": {
                        "behavior": "allow",
                        "updatedInput": request.get("input", {}),
                        "toolUseID": request.get("tool_use_id"),
                    },
                },
            }
        else:
            resp = {
                "type": "control_response",
                "response": {
                    "subtype": "error",
                    "request_id": rid,
                    "error": f"unsupported control_request subtype: {subtype}",
                },
            }
        self._write(proc, resp)
        return stop

    def _handle_mcp(
        self, message: dict, tools: list[dict], on_tool_call: ToolHandler
    ) -> "tuple[dict, bool]":
        mid = message.get("id")
        method = message.get("method", "")
        params = message.get("params") or {}

        if method == "initialize":
            return {
                "jsonrpc": "2.0", "id": mid,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": _SERVER_NAME, "version": "0.1.0"},
                },
            }, False
        if method == "notifications/initialized":
            return {"jsonrpc": "2.0", "result": {}}, False
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": mid, "result": {"tools": tools}}, False
        if method == "tools/call":
            name = params.get("name", "")
            args = params.get("arguments") or {}
            output, is_error, stop = on_tool_call(name, args)
            return {
                "jsonrpc": "2.0", "id": mid,
                "result": {"content": [{"type": "text", "text": output}], "isError": bool(is_error)},
            }, stop
        return {
            "jsonrpc": "2.0", "id": mid,
            "error": {"code": -32601, "message": f"unsupported MCP method '{method}'"},
        }, False

    @staticmethod
    def _write(proc: subprocess.Popen, value: dict) -> None:
        line = json.dumps(value, ensure_ascii=False)
        assert proc.stdin is not None
        proc.stdin.write(line + "\n")
        proc.stdin.flush()

    @staticmethod
    def _shutdown(proc: subprocess.Popen) -> None:
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
