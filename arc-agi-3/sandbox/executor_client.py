from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import threading
import time
from typing import Any, Optional

from agent.world_model.budgets import JAIL_BFS_SEARCH_S, JAIL_STEP_CHECK_S, JAIL_TOOL_CAP_S

from .paths import JAIL_SRC


class ExecutorError(RuntimeError):
    pass


class ExecutorWedged(ExecutorError):
    pass


class ExecutorClient:
    def __init__(
        self,
        *,
        runtime,
        container: str,
        mounts: "list[tuple]",
        jail_env: dict[str, str],
        ro_mounts: list[str],
        allow_only: list[str],
        workdir: str,
        log=None,
        rpc_timeout: float = 360.0,
    ) -> None:
        self.runtime = runtime
        self.container = container
        self.mounts = mounts
        self.jail_env = jail_env
        self.ro_mounts = ro_mounts
        self.allow_only = allow_only
        self.workdir = workdir
        self._log = log or (lambda m: print(f"[executor] {m}", file=sys.stderr, flush=True))
        self._rpc_timeout = rpc_timeout
        self._proc: Optional[subprocess.Popen] = None
        self._next_id = 0
        self._lock = threading.Lock()
        self._wfd: Optional[int] = None
        self._wedged = False
        self._last_world_code: Optional[str] = None

    def _build_cmd(self) -> list[str]:
        record_mounts = self.runtime.record_ro(self.ro_mounts)
        argv = ["python3", "-u", f"{JAIL_SRC}/sandbox/agent_executor.py",
                "--workdir", self.workdir, "--allow-only", ":".join(self.allow_only)]
        return self.runtime.container_cmd(
            name=self.container, mounts=list(self.mounts) + record_mounts,
            env=self.jail_env, argv=argv)

    def start(self) -> None:
        self.runtime.create(self.container)
        cmd = self._build_cmd()
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, env=self.runtime.popen_env, close_fds=True,
        )
        assert self._proc.stdin
        self._wfd = self._proc.stdin.fileno()
        os.set_blocking(self._wfd, False)
        self._wedged = False
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        pong = self.call("ping", _timeout=60.0)
        self._log(f"executor ready: uid={pong.get('uid')} cwd={pong.get('cwd')}")

    def restart(self) -> None:
        code = self._last_world_code
        self._log("executor channel is wedged -> restarting a fresh executor")
        try:
            self.close()
        except Exception:
            pass
        self.start()
        if code:
            self._last_world_code = code
            try:
                self.call("world_load", code=code, _timeout=60)
                self._log("world model reloaded after restart")
            except Exception as e:
                self._log(f"failed to reload the world model after restart: {e}")

    def _write_all(self, payload: str, timeout: float) -> None:
        assert self._wfd is not None
        data = payload.encode("utf-8")
        deadline = time.monotonic() + timeout
        while data:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._wedged = True
                raise ExecutorWedged(
                    f"executor stopped reading stdin ({timeout:.0f}s write timeout); restart needed")
            _, wl, _ = select.select([], [self._wfd], [], remaining)
            if not wl:
                continue
            try:
                n = os.write(self._wfd, data)
            except BlockingIOError:
                continue
            except OSError as e:
                self._wedged = True
                raise ExecutorWedged(f"writing executor stdin failed: {e}") from e
            data = data[n:]

    def _drain_stderr(self) -> None:
        assert self._proc and self._proc.stderr
        try:
            for line in self._proc.stderr:
                self._log(line.rstrip())
        except Exception:
            pass

    def call(self, op: str, *, _timeout: Optional[float] = None, **kwargs: Any) -> dict:
        if self._wedged:
            self.restart()
        if self._proc is None or self._proc.poll() is not None:
            raise ExecutorError("executor process is not running")
        deadline_s = _timeout if _timeout is not None else self._rpc_timeout
        with self._lock:
            self._next_id += 1
            req_id = self._next_id
            req = {"id": req_id, "op": op, **kwargs}
            assert self._proc.stdout
            self._write_all(json.dumps(req, ensure_ascii=False) + "\n", timeout=deadline_s)
            deadline = time.monotonic() + deadline_s
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._wedged = True
                    raise ExecutorError(
                        f"executor RPC timeout (op={op}, {deadline_s:.0f}s); channel marked for restart")
                rl, _, _ = select.select([self._proc.stdout], [], [], remaining)
                if not rl:
                    continue
                line = self._proc.stdout.readline()
                if not line:
                    raise ExecutorError(f"executor closed the pipe (op={op})")
                try:
                    resp = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rid = resp.get("id")
                if rid is not None and rid != req_id:
                    self._log(f"dropping stale response id={rid} (current request id={req_id})")
                    continue
                break
        if not resp.get("ok"):
            raise ExecutorError(f"executor op {op} failed: {resp.get('error')}")
        return resp.get("result") or {}

    def run_python(self, *, code: Optional[str] = None, path: Optional[str] = None,
                   args: Optional[list] = None, stdin: Optional[str] = None,
                   timeout: float = 300.0) -> dict:
        return self.call("run_python", code=code, path=path, args=args or [], stdin=stdin,
                         timeout=timeout, _timeout=timeout + 30)

    def run_shell(self, command: str, *, cwd: Optional[str] = None,
                  timeout: float = 300.0) -> dict:
        return self.call("run_shell", command=command, cwd=cwd,
                         timeout=timeout, _timeout=timeout + 30)

    def world_load(self, code: str) -> dict:
        r = self.call("world_load", code=code, _timeout=60)
        if r.get("loaded"):
            self._last_world_code = code
        return r

    def world_set_entry(self, *, grid, level) -> dict:
        return self.call("world_set_entry", grid=grid, level=level)

    def world_predict_step(self, *, timeline, entry, level, before_grid, action, x=None, y=None,
                           want_state: bool = False) -> dict:
        return self.call("world_predict_step", timeline=timeline, entry=entry, level=level,
                         before_grid=before_grid, action=action, x=x, y=y,
                         want_state=bool(want_state), _timeout=JAIL_STEP_CHECK_S)

    def world_bfs(self, *, grid, acts, clicks, target, timeline, entry, level,
                  max_depth, max_nodes, allow_reset) -> dict:
        time_budget_s = JAIL_BFS_SEARCH_S
        deadline = max(JAIL_TOOL_CAP_S, time_budget_s + 30.0)
        return self.call("world_bfs", grid=grid, acts=acts, clicks=clicks, target=target,
                         timeline=timeline, entry=entry, level=level, max_depth=max_depth,
                         max_nodes=max_nodes, allow_reset=allow_reset,
                         time_budget_s=time_budget_s, _timeout=deadline)

    def world_backtest(self, *, timeline, entry, code=None) -> dict:
        return self.call("world_backtest", timeline=timeline, entry=entry, code=code,
                         _timeout=JAIL_TOOL_CAP_S)

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin and not self._proc.stdin.closed:
                self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
        self.runtime.cleanup(self.container)
