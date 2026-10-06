from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from agent.world_model.run_records import RECORD_FILES, RECORD_NAMES, record_paths

from .claude_launch import ClaudeContainerLaunch
from .executor_client import ExecutorClient
from .paths import JAIL_SRC, PROXY_PORT, STAGED_TOPS
from .runtime import make_runtime

_REPO = Path(__file__).resolve().parents[1]

_ALLOW = [".anthropic.com", ".claude.ai", ".claude.com"]


def _stage_src(stage: Path) -> None:
    if stage.exists():
        shutil.rmtree(stage)
    ign = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
    for top in STAGED_TOPS:
        shutil.copytree(_REPO / top, stage / top, ignore=ign)


class SandboxOrchestrator:
    def __init__(self, *, workdir: "str | Path", runtime: str = "podman", log=None) -> None:
        self.workdir = Path(workdir).resolve()
        self.runtime_name = runtime
        self._log = log or (lambda m: print(f"[orchestrator] {m}", file=sys.stderr, flush=True))
        self._proxy: "subprocess.Popen | None" = None
        self.executor: "ExecutorClient | None" = None
        self.runtime = None
        self._exec_ctr = f"arc-v2-exec-{os.getpid()}"
        self._claude_ctr = f"arc-v2-claude-{os.getpid()}"
        self._proxy_ctr = f"arc-v2-proxy-{os.getpid()}"
        self._net = f"arc-v2-net-{os.getpid()}"
        self._local_rt = Path(os.environ.get("TMPDIR", "/tmp")) / f"arcjail-{os.getpid()}"

    def start(self) -> "tuple[ClaudeContainerLaunch, ExecutorClient]":
        sockdir = self._local_rt / "sock"
        sockdir.mkdir(parents=True, exist_ok=True)
        rt = self.workdir.parent / (self.workdir.name + ".rt")
        logs = rt / "logs"
        stage = rt / "src"
        logs.mkdir(parents=True, exist_ok=True)
        _stage_src(stage)

        enroot_env = dict(
            os.environ,
            ENROOT_DATA_PATH=str(self._local_rt / "enroot" / "data"),
            ENROOT_CACHE_PATH=str(self._local_rt / "enroot" / "cache"),
            ENROOT_RUNTIME_PATH=str(self._local_rt / "enroot" / "run"),
            ENROOT_TEMP_PATH=str(self._local_rt / "enroot" / "tmp"),
        )
        if self.runtime_name == "enroot":
            for k in ("ENROOT_DATA_PATH", "ENROOT_CACHE_PATH", "ENROOT_RUNTIME_PATH", "ENROOT_TEMP_PATH"):
                os.makedirs(enroot_env[k], exist_ok=True)

        self.runtime = make_runtime(self.runtime_name, enroot_env=enroot_env)
        image = self.runtime.ensure_image()
        self._log(f"runtime={self.runtime.name} image={image}")

        proxy_ctr_mode = self.runtime.proxy_in_container
        upstream = os.environ.get("ARC_SANDBOX_UPSTREAM_PROXY") or None
        upstream_args = ["--upstream", upstream] if upstream else []
        if proxy_ctr_mode:
            self.runtime.network_create(self._net)
            dns = [d for d in os.environ.get("ARC_SANDBOX_PROXY_DNS", "1.1.1.1,8.8.8.8").split(",") if d]
            proxy_cmd = self.runtime.cli(
                "create", "--rm", "--name", self._proxy_ctr,
                *self.runtime._identity_args(),
                "--add-host", "host.docker.internal:host-gateway",
                *[a for d in dns for a in ("--dns", d)],
                "-v", f"{stage}:{JAIL_SRC}:ro", "-e", f"HOME={self.runtime.container_home}",
                image, "python3", "-u",
                f"{JAIL_SRC}/sandbox/proxy_server.py",
                "--tcp-port", str(PROXY_PORT), "--allow", ",".join(_ALLOW), *upstream_args)
            r = subprocess.run(proxy_cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"{self.runtime.name} create proxy failed: {r.stderr.strip()[:300]}")
            subprocess.run(self.runtime.cli("network", "connect", self._net, self._proxy_ctr),
                           capture_output=True, check=True)
            self._proxy = subprocess.Popen(
                self.runtime.cli("start", "-a", self._proxy_ctr),
                stdout=open(logs / "proxy.log", "a", buffering=1), stderr=subprocess.STDOUT,
            )
            for _ in range(100):
                ok = subprocess.run(
                    self.runtime.cli("exec", self._proxy_ctr, "python3", "-c",
                     f'import socket; socket.create_connection(("127.0.0.1",{PROXY_PORT}),0.2).close()'),
                    capture_output=True)
                if ok.returncode == 0:
                    break
                time.sleep(0.2)
        else:
            proxy_sock = sockdir / "proxy.sock"
            self._proxy = subprocess.Popen(
                [sys.executable, str(Path(__file__).parent / "proxy_server.py"),
                 "--socket", str(proxy_sock), "--allow", ",".join(_ALLOW), *upstream_args],
                stdout=open(logs / "proxy.log", "a", buffering=1), stderr=subprocess.STDOUT,
            )
            for _ in range(100):
                if proxy_sock.exists():
                    break
                time.sleep(0.1)
        self._log(f"proxy allowlist: {', '.join(_ALLOW)}")

        self.runtime.create(self._claude_ctr)

        wd = str(self.workdir)
        for name in RECORD_NAMES:
            p = self.workdir / name
            if not p.exists():
                if name in RECORD_FILES:
                    p.touch()
                else:
                    p.mkdir(parents=True)
        exec_mounts = [(str(stage), JAIL_SRC, "ro"), (wd, wd, "rw")]
        exec_env = {"HOME": self.runtime.container_home,
                    "PATH": "/usr/local/bin:/usr/local/sbin:/usr/bin:/usr/sbin:/bin:/sbin",
                    "PYTHONPATH": JAIL_SRC, "TMPDIR": "/tmp", "PYTHONUNBUFFERED": "1"}
        self.executor = ExecutorClient(
            runtime=self.runtime, container=self._exec_ctr, mounts=exec_mounts,
            jail_env=exec_env, ro_mounts=record_paths(self.workdir),
            allow_only=["/usr", "/etc", JAIL_SRC], workdir=wd, log=self._log)
        self.executor.start()

        launch = ClaudeContainerLaunch(
            container=self._claude_ctr, runtime=self.runtime, stage_src=str(stage),
            proxy_sock_dir=str(sockdir), workdir=wd,
            proxy_host=self._proxy_ctr if proxy_ctr_mode else None,
            network=self._net if proxy_ctr_mode else None)
        self._log("orchestrator ready (harness outside; claude + run_python in containers)")
        return launch, self.executor

    def stop(self) -> None:
        if self.executor is not None:
            try:
                self.executor.close()
            except Exception:
                pass
        rt = self.runtime
        if rt is not None:
            for ctr in (self._claude_ctr, self._exec_ctr):
                rt.cleanup(ctr)
            if rt.proxy_in_container:
                rt.cleanup(self._proxy_ctr)
        if self._proxy is not None and self._proxy.poll() is None:
            self._proxy.terminate()
            try:
                self._proxy.wait(timeout=5)
            except Exception:
                self._proxy.kill()
        if rt is not None and rt.proxy_in_container:
            rt.network_rm(self._net)
        shutil.rmtree(self._local_rt, ignore_errors=True)
        self._log("orchestrator stopped")
