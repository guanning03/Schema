from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Optional

from . import call_clock

ToolHandler = Callable[[str, dict], "tuple[str, bool, bool]"]
MessageBuilder = Callable[[bool], dict]

_FALLBACK_LIMIT_COOLDOWN = 300.0
_TOOL_BUDGET = 100
_TURN_IDLE_TIMEOUT = 1200.0
_LIMIT_RE = re.compile(
    r"usage limit|rate.?limit|quota|too many requests|\b429\b", re.I
)
_RETRY_IN_RE = re.compile(
    r"try again in\s+(?:(\d+)\s*hours?)?\s*(?:(\d+)\s*minutes?)?\s*(?:(\d+)\s*seconds?)?", re.I
)

_AUDIT_ITEM_TYPES = ("web_search", "command_execution", "patch", "file_change")

_BILLING_DIVERTING_ENV = ("OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL")

_HARNESS_RULES = """
## Harness rules (appended by the driver)
- The ONLY tools you may use are the MCP tools from the `arc` server. Their names may be shown with an `mcp__arc__` prefix: `mcp__arc__X` is the same tool as `X` referenced in the instructions above. This INCLUDES your workspace file/exec tools (e.g. read_file, write_file, run_python, run_shell) - they run in YOUR workspace and are always allowed.
- NEVER use codex's built-in capabilities: web search/browsing, apply_patch, update_plan, view_image, request_user_input, plan/todo tools, or the built-in terminal. They are disabled or sandboxed; attempting them wastes the turn. (This prohibition does NOT apply to the mcp__arc__* tools.)
- After a successful commit_actions call, END YOUR TURN immediately: reply with one short line and stop. Any further tool call after commit will fail with an error.
"""


@dataclass
class TurnResult:
    final_text: str = ""
    usage: dict = field(default_factory=dict)
    committed: bool = False
    thread_id: Optional[str] = None
    stderr: str = ""
    error_kind: Optional[str] = None
    limit_reset_at: Optional[float] = None


def _parse_retry_in(blob: str) -> Optional[float]:
    m = _RETRY_IN_RE.search(blob)
    if not m or not any(m.groups()):
        return None
    h, mi, s = (int(g) if g else 0 for g in m.groups())
    secs = h * 3600 + mi * 60 + s
    return time.time() + secs if secs > 0 else None


_RETRY_AT_RE = re.compile(
    r"try again at\s+([A-Za-z]{3})[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\s+(\d{1,2}):(\d{2})\s*([AP]M)", re.I
)
_RESET_MARKER = ".reset_card_consumed"


def _parse_retry_at(blob: str) -> Optional[float]:
    m = _RETRY_AT_RE.search(blob)
    if not m:
        return None
    mon, day, year, hh, mm, ampm = m.groups()
    try:
        return time.mktime(time.strptime(f"{mon.title()} {day} {year} {hh}:{mm} {ampm.upper()}",
                                         "%b %d %Y %I:%M %p"))
    except (ValueError, OverflowError):
        return None


def _reset_since(account_dir: str, t: float) -> bool:
    try:
        return os.path.getmtime(os.path.join(account_dir, _RESET_MARKER)) > t
    except OSError:
        return False


class CodexAccountPool:

    def __init__(self, dirs: list[str], notify: Optional[Callable[[str], None]] = None) -> None:
        self._dirs = list(dirs)
        self._cooldown: dict[str, float] = {}
        self._marked: dict[str, float] = {}
        self._idx = 0
        self._lock = threading.Lock()
        self._notify = notify or (lambda m: print(m, file=sys.stderr, flush=True))

    def __len__(self) -> int:
        return len(self._dirs)

    @classmethod
    def discover(cls, notify: Optional[Callable[[str], None]] = None) -> "Optional[CodexAccountPool]":
        raw = os.environ.get("ARC_CODEX_POOL", "").strip()
        if raw:
            dirs = [os.path.expanduser(p) for p in re.split(r"[:,]", raw) if p.strip()]
        else:
            dirs = sorted(str(p) for p in Path.home().glob(".codex-arc-agent*") if p.is_dir())
        dirs = [d for d in dirs if os.path.isfile(os.path.join(d, "auth.json"))]
        if not dirs:
            return None
        return cls(dirs, notify=notify)

    def acquire(self) -> str:
        while True:
            now = time.time()
            with self._lock:
                for d in list(self._cooldown):
                    if self._cooldown[d] <= now or _reset_since(d, self._marked.get(d, now)):
                        del self._cooldown[d]
                for d in self._dirs:
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
        with self._lock:
            self._cooldown[account_dir] = reset_at
            self._marked[account_dir] = time.time()
            self._idx = (self._idx + 1) % len(self._dirs)
        when = time.strftime("%H:%M:%S", time.localtime(reset_at))
        self._notify(f"[codex-pool] {account_dir} hit {kind} (expected reset {when}), switching account.")


class _TurnState:

    def __init__(self, tools: list[dict], on_tool_call: ToolHandler,
                 tool_budget: int = _TOOL_BUDGET) -> None:
        self.tools = tools
        self.on_tool_call = on_tool_call
        self.committed = False
        self.lock = threading.Lock()
        self.in_tool = False
        self.last_activity = time.time()
        self.jobs: "Optional[queue.Queue]" = None
        self.tool_budget = tool_budget
        self.tool_calls = 0
        self.closed = False


class _ToolJob:
    def __init__(self, name: str, args: dict) -> None:
        self.name = name
        self.args = args
        self.done = threading.Event()
        self.sent_at = time.monotonic()
        self.abandoned = False
        self.result: dict = {"content": [{"type": "text", "text": "turn ended before tool ran"}],
                             "isError": True}


class _McpHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    driver: "CodexCliDriver"
    mcp_session_id: str


class _McpHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send_json(self, obj) -> None:
        payload = json.dumps(obj).encode("utf-8")
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Mcp-Session-Id", self.server.mcp_session_id)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except OSError:
            pass

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
            msg = json.loads(self.rfile.read(length))
        except Exception:
            self.send_response(400)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if isinstance(msg, list):
            resps = [r for r in (self.server.driver._handle_mcp(m) for m in msg) if r is not None]
            if resps:
                self._send_json(resps)
                return
            resp = None
        else:
            resp = self.server.driver._handle_mcp(msg)
        if resp is None:
            self.send_response(202)
            self.send_header("Content-Length", "0")
            self.end_headers()
        else:
            self._send_json(resp)

    def do_GET(self):
        self.send_response(405)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_DELETE(self):
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()


class CodexCliDriver:

    def __init__(
        self,
        *,
        model: str,
        reasoning: Optional[str] = None,
        cwd: Optional[str] = None,
        service_tier: Optional[str] = None,
    ) -> None:
        self.model = model
        self.reasoning = reasoning
        # Codex service tier ("ultrafast", "priority" = Fast). Plans that do not offer the tier
        # get standard speed: Codex drops it from the request and reports that once per turn.
        self.service_tier = service_tier or None
        self._tier_warned = False
        self.cwd = Path(cwd or os.getcwd()).resolve()
        self._notify = lambda m: print(m, file=sys.stderr, flush=True)
        self._exe = shutil.which("codex")

        self.pool = CodexAccountPool.discover(self._notify)
        if self.pool is not None:
            self._notify(f"[codex-cli] account pool: {len(self.pool)} account(s)")

        base = Path(os.environ.get("TMPDIR") or "/tmp")
        self.codex_home = base / f"codex_home-{os.getpid()}-{abs(hash(str(self.cwd))) % 10**8}"
        self.codex_home.mkdir(parents=True, exist_ok=True)
        self.sandbox_cwd = self.cwd / "codex_cwd"
        self.sandbox_cwd.mkdir(parents=True, exist_ok=True)
        self._ensure_gitignored(("codex_home/", "codex_cwd/"))
        self._instr_path = self.codex_home / "instructions.md"
        self._instr_hash: Optional[str] = None

        self._thread_id: Optional[str] = None
        self._turn_no = 0
        self._state: Optional[_TurnState] = None
        self._state_lock = threading.Lock()
        self._default_auth_src: Optional[Path] = None

        self._http = _McpHTTPServer(("127.0.0.1", 0), _McpHandler)
        self._http.driver = self
        self._http.mcp_session_id = uuid.uuid4().hex
        threading.Thread(target=self._http.serve_forever, daemon=True).start()
        self._mcp_port = self._http.server_address[1]

        self._write_config()
        if self.pool is None:
            self._seed_default_auth()

    def available(self) -> bool:
        if self._exe is None:
            return False
        if self.pool is not None:
            return True
        return (self.codex_home / "auth.json").is_file()

    def _ensure_gitignored(self, entries: "tuple[str, ...]") -> None:
        gi = self.cwd / ".gitignore"
        try:
            existing = gi.read_text(encoding="utf-8").splitlines() if gi.is_file() else []
            missing = [e for e in entries if e not in existing]
            if missing:
                gi.write_text("\n".join(existing + missing) + "\n", encoding="utf-8")
        except OSError:
            pass

    def _seed_default_auth(self) -> None:
        src = Path(os.path.expanduser("~/.codex/auth.json"))
        if src.is_file():
            self._default_auth_src = src
        dst = self.codex_home / "auth.json"
        if dst.is_file():
            return
        if src.is_file():
            shutil.copy2(src, dst)

    def _syncback_default_auth(self) -> None:
        src = self.codex_home / "auth.json"
        dst = self._default_auth_src
        if dst is None or not src.is_file():
            return
        try:
            if not dst.is_file() or src.stat().st_mtime > dst.stat().st_mtime:
                shutil.copy2(src, dst)
        except OSError:
            pass

    def _install_auth(self, account_dir: str) -> None:
        src = Path(account_dir) / "auth.json"
        if src.is_file():
            shutil.copy2(src, self.codex_home / "auth.json")

    def _syncback_auth(self, account_dir: str) -> None:
        src = self.codex_home / "auth.json"
        dst = Path(account_dir) / "auth.json"
        try:
            if src.is_file() and (not dst.is_file() or src.stat().st_mtime > dst.stat().st_mtime):
                shutil.copy2(src, dst)
        except OSError:
            pass

    def _write_config(self) -> None:
        cfg = f"""model = "{self.model}"
approval_policy = "never"
model_instructions_file = "{self._instr_path}"
web_search = "disabled"

[features]
shell_tool = false
unified_exec = false
plugins = false
image_generation = false
apps = false
multi_agent = false
goals = false

[tools]
view_image = false

[mcp_servers.arc]
url = "http://127.0.0.1:{self._mcp_port}/mcp"
default_tools_approval_mode = "approve"
startup_timeout_sec = 30
tool_timeout_sec = {self._CODEX_TOOL_TIMEOUT_S}

[projects."{self.sandbox_cwd}"]
trust_level = "trusted"
"""
        (self.codex_home / "config.toml").write_text(cfg, encoding="utf-8")

    def _ensure_instructions(self, system_prompt: str) -> None:
        text = system_prompt.rstrip() + "\n" + _HARNESS_RULES
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest != self._instr_hash:
            self._instr_path.write_text(text, encoding="utf-8")
            self._instr_hash = digest

    @staticmethod
    def _to_mcp_tools(tools: list[dict]) -> list[dict]:
        return [
            {
                "name": t["name"],
                "description": t.get("description", ""),
                "inputSchema": t["input_schema"],
                "annotations": {
                    "readOnlyHint": True,
                    "destructiveHint": False,
                    "openWorldHint": False,
                },
            }
            for t in tools
        ]

    def _handle_mcp(self, msg: dict) -> Optional[dict]:
        if not isinstance(msg, dict):
            return None
        mid = msg.get("id")
        method = msg.get("method", "")
        params = msg.get("params") or {}

        if method == "initialize":
            return {
                "jsonrpc": "2.0", "id": mid,
                "result": {
                    "protocolVersion": params.get("protocolVersion", "2025-06-18"),
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "arc", "version": "0.1.0"},
                },
            }
        if method.startswith("notifications/"):
            return None
        with self._state_lock:
            state = self._state
        if method == "tools/list":
            tools = self._to_mcp_tools(state.tools) if state else []
            return {"jsonrpc": "2.0", "id": mid, "result": {"tools": tools}}
        if method == "tools/call":
            return {"jsonrpc": "2.0", "id": mid, "result": self._dispatch_tool(state, params)}
        if mid is not None:
            return {"jsonrpc": "2.0", "id": mid,
                    "error": {"code": -32601, "message": f"unsupported MCP method '{method}'"}}
        return None

    _CODEX_TOOL_TIMEOUT_S = 1680
    _TOOL_JOB_WAIT = 1800.0

    def _dispatch_tool(self, state: Optional[_TurnState], params: dict) -> dict:
        name = params.get("name", "")
        args = params.get("arguments") or {}
        if not isinstance(args, dict):
            args = {}
        if state is None or state.closed:
            return {"content": [{"type": "text", "text": "no active turn"}], "isError": True}
        jobs = state.jobs
        if jobs is None:
            return self._execute_tool(state, name, args)
        job = _ToolJob(name, args)
        jobs.put(("tool", job))
        if not job.done.wait(self._TOOL_JOB_WAIT):
            job.abandoned = True
            return {"content": [{"type": "text", "text": "tool execution timed out in driver"}],
                    "isError": True}
        return job.result

    def _run_job(self, state: _TurnState, job: "_ToolJob") -> None:
        if not (job.abandoned or time.monotonic() - job.sent_at >= self._CODEX_TOOL_TIMEOUT_S):
            call_clock.set_sent_at(job.sent_at)
            try:
                job.result = self._execute_tool(state, job.name, job.args)
            finally:
                call_clock.set_sent_at(None)
        job.done.set()

    def _execute_tool(self, state: _TurnState, name: str, args: dict) -> dict:
        with state.lock:
            if state.committed:
                return {
                    "content": [{"type": "text", "text":
                                 "Turn already committed. STOP: do not call any more tools; "
                                 "end your turn with a one-line summary."}],
                    "isError": True,
                }
            if state.tool_budget > 0 and state.tool_calls >= state.tool_budget:
                return {
                    "content": [{"type": "text", "text":
                                 f"Tool budget for this turn is exhausted ({state.tool_budget} "
                                 "calls). END YOUR TURN NOW: reply with a one-line summary. "
                                 "You will get a fresh turn with the current observation."}],
                    "isError": True,
                }
            state.tool_calls += 1
            state.in_tool = True
            try:
                output, is_error, want_stop = state.on_tool_call(name, args)
            except Exception as e:
                state.in_tool = False
                state.last_activity = time.time()
                return {"content": [{"type": "text", "text": f"tool crashed: {type(e).__name__}: {e}"}],
                        "isError": True}
            state.in_tool = False
            state.last_activity = time.time()
            if want_stop:
                state.committed = True
                output += ("\n\nActions committed and executed. END YOUR TURN NOW - reply with "
                           "one short line and stop; any further tool call will fail.")
        return {"content": [{"type": "text", "text": output}], "isError": bool(is_error)}

    def run_turn(
        self,
        system_prompt: str,
        build_user_message: MessageBuilder,
        tools: list[dict],
        on_tool_call: ToolHandler,
    ) -> TurnResult:
        if self._exe is None:
            raise RuntimeError("`codex` CLI not found on PATH. npm i -g @openai/codex")
        self._ensure_instructions(system_prompt)
        self._turn_no += 1

        stall_retries = 0
        while True:
            account_dir = self.pool.acquire() if self.pool is not None else None
            if account_dir:
                self._install_auth(account_dir)
            result = self._run_once(build_user_message, tools, on_tool_call)
            if account_dir:
                self._syncback_auth(account_dir)
            else:
                self._syncback_default_auth()

            if result.error_kind == "limit" and not result.committed:
                if self.pool is not None:
                    self.pool.mark_limited(account_dir, "limit", result.limit_reset_at)
                    continue
                wake = result.limit_reset_at or (time.time() + _FALLBACK_LIMIT_COOLDOWN)
                self._notify(
                    f"[codex-cli] hit the usage limit, sleeping until "
                    f"{time.strftime('%H:%M:%S', time.localtime(wake))} before retrying this turn…"
                )
                while time.time() < wake:
                    time.sleep(min(60.0, max(1.0, wake - time.time())))
                continue
            if result.error_kind == "stalled" and not result.committed and stall_retries < 3:
                stall_retries += 1
                self._notify(f"[codex-cli] no output, process killed; retrying this turn ({stall_retries}/3)")
                continue
            if result.error_kind == "error":
                tail = " | ".join((result.stderr or "").strip().splitlines()[-3:])
                self._notify(f"[codex-cli] turn failed (error): {tail or 'no stderr output'}")
            return result

    def _build_cmd(self, resume_tid: Optional[str]) -> list[str]:
        cmd = [self._exe, "-m", self.model]
        if self.reasoning:
            cmd += ["-c", f"model_reasoning_effort={self.reasoning}"]
        if self.service_tier:
            cmd += ["-c", f"service_tier={self.service_tier}"]
        cmd += ["--disable", "shell_tool", "--disable", "plugins"]
        cmd += ["exec", "--json", "--color", "never",
                "--skip-git-repo-check", "-s", "read-only", "--cd", str(self.sandbox_cwd)]
        if resume_tid:
            cmd += ["resume", resume_tid]
        cmd.append("-")
        return cmd

    @staticmethod
    def _render_user_message(user_message: dict) -> str:
        content = user_message.get("content", "")
        if isinstance(content, str):
            return content
        texts = [blk.get("text", "") for blk in content
                 if isinstance(blk, dict) and blk.get("type") == "text"]
        return "\n\n".join(t for t in texts if t)

    def _run_once(
        self,
        build_user_message: MessageBuilder,
        tools: list[dict],
        on_tool_call: ToolHandler,
    ) -> TurnResult:
        resume_tid = self._thread_id
        user_message = build_user_message(resume_tid is not None)
        prompt_text = self._render_user_message(user_message)
        if not prompt_text.strip():
            prompt_text = "(continue)"

        state = _TurnState(tools, on_tool_call)
        jobs: "queue.Queue" = queue.Queue()
        state.jobs = jobs
        with self._state_lock:
            self._state = state

        env = dict(os.environ)
        env["CODEX_HOME"] = str(self.codex_home)
        env.setdefault("RUST_LOG", "error")
        for k in _BILLING_DIVERTING_ENV:
            env.pop(k, None)

        log_fh = (self.codex_home / "exec_events.jsonl").open("a", encoding="utf-8")

        proc = subprocess.Popen(
            self._build_cmd(resume_tid),
            cwd=str(self.sandbox_cwd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )

        err_buf: list[str] = []

        def _drain_err() -> None:
            try:
                for line in proc.stderr:
                    err_buf.append(line)
            except Exception:
                pass

        threading.Thread(target=_drain_err, daemon=True).start()

        def _drain_out() -> None:
            try:
                for line in proc.stdout:
                    jobs.put(("line", line))
            except Exception:
                pass
            finally:
                jobs.put(("eof", None))

        threading.Thread(target=_drain_out, daemon=True).start()

        stalled = False

        result = TurnResult(thread_id=resume_tid)
        error_blobs: list[str] = []
        turn_completed = False
        turn_failed = False
        try:
            try:
                assert proc.stdin is not None
                proc.stdin.write(prompt_text)
                proc.stdin.close()
            except OSError:
                pass

            while True:
                try:
                    kind, payload = jobs.get(timeout=3.0)
                except queue.Empty:
                    if not stalled and time.time() - state.last_activity > _TURN_IDLE_TIMEOUT:
                        stalled = True
                        try:
                            proc.kill()
                        except Exception:
                            pass
                    continue
                if kind == "eof":
                    break
                if kind == "tool":
                    job = payload
                    self._run_job(state, job)
                    state.last_activity = time.time()
                    continue
                state.last_activity = time.time()
                try:
                    log_fh.write(payload if payload.endswith("\n") else payload + "\n")
                    log_fh.flush()
                except OSError:
                    pass
                line = payload.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                etype = ev.get("type", "")
                if etype == "thread.started":
                    tid = ev.get("thread_id")
                    if isinstance(tid, str) and tid:
                        result.thread_id = tid
                elif etype == "item.completed":
                    item = ev.get("item") or {}
                    itype = item.get("type", "")
                    if itype == "agent_message":
                        result.final_text = item.get("text") or ""
                    elif itype == "mcp_tool_call":
                        if item.get("status") == "failed":
                            error_blobs.append(json.dumps(item.get("error") or {}))
                    elif itype == "error" and "service tier" in str(item.get("message", "")):
                        if not self._tier_warned:
                            self._tier_warned = True
                            self._notify(f"[codex-cli] {item.get('message')} (continuing at standard speed)")
                    if any(k in itype for k in _AUDIT_ITEM_TYPES):
                        try:
                            with (self.cwd / "codex_audit.jsonl").open("a", encoding="utf-8") as af:
                                af.write(json.dumps({"turn": self._turn_no, "item": item},
                                                    ensure_ascii=False) + "\n")
                        except OSError:
                            pass
                        self._notify(
                            f"[codex-cli][audit] the model used a disabled built-in capability: {itype} "
                            f"— this run may violate the no-external-help constraint. "
                            f"item={json.dumps(item)[:200]}"
                        )
                elif etype == "turn.completed":
                    turn_completed = True
                    result.usage = ev.get("usage") or {}
                elif etype in ("turn.failed", "error"):
                    turn_failed = True
                    error_blobs.append(json.dumps(ev))
        finally:
            try:
                proc.wait(timeout=10)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            with self._state_lock:
                self._state = None
            state.closed = True
            while True:
                try:
                    kind, payload = jobs.get_nowait()
                except queue.Empty:
                    break
                if kind == "tool":
                    payload.done.set()
            result.committed = state.committed
            result.stderr = "".join(err_buf)
            try:
                if proc.returncode not in (0, None) and result.stderr.strip():
                    tail = result.stderr.strip().splitlines()[-5:]
                    log_fh.write(json.dumps({"type": "driver.note", "returncode": proc.returncode,
                                             "stderr_tail": tail}, ensure_ascii=False) + "\n")
                log_fh.close()
            except OSError:
                pass

        blob = "\n".join([result.stderr, result.final_text, *error_blobs])
        hard_failed = proc.returncode not in (0, None) or turn_failed
        if hard_failed:
            if _LIMIT_RE.search(blob):
                result.error_kind = "limit"
                result.limit_reset_at = _parse_retry_in(blob) or _parse_retry_at(blob)
            else:
                result.error_kind = "error"
        if stalled and not state.committed:
            result.error_kind = "stalled"
        if result.committed:
            result.error_kind = None

        if result.committed or (result.error_kind is None and turn_completed and result.thread_id):
            self._thread_id = result.thread_id or self._thread_id
        elif result.error_kind == "error" and resume_tid is not None:
            self._thread_id = None
        return result

    def _find_rollout(self, tid: str) -> Optional[Path]:
        sessions = self.codex_home / "sessions"
        if not sessions.is_dir():
            return None
        hits = sorted(sessions.rglob(f"*{tid}*.jsonl"))
        return hits[-1] if hits else None

    def export_sessions(self, dest: Path) -> None:
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        for old in dest.glob("rollout-*.jsonl"):
            try:
                old.unlink()
            except OSError:
                pass
        tid = self._thread_id
        relpath: Optional[str] = None
        if tid:
            rollout = self._find_rollout(tid)
            if rollout is not None:
                relpath = str(rollout.relative_to(self.codex_home / "sessions"))
                shutil.copy2(rollout, dest / rollout.name)
            else:
                tid = None
        (dest / "codex_thread.json").write_text(
            json.dumps({"thread_id": tid, "rollout_relpath": relpath}), encoding="utf-8"
        )

    def import_sessions(self, src: Path) -> None:
        src = Path(src)
        meta_path = src / "codex_thread.json"
        self._thread_id = None
        if not meta_path.is_file():
            return
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8")) or {}
        except (OSError, ValueError):
            return
        tid, relpath = meta.get("thread_id"), meta.get("rollout_relpath")
        if not tid or not relpath:
            return
        dump = src / Path(relpath).name
        if not dump.is_file():
            return
        dst = self.codex_home / "sessions" / relpath
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dump, dst)
        except OSError:
            return
        self._thread_id = tid
