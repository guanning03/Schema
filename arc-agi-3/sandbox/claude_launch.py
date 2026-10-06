from __future__ import annotations

import os
import shlex
from pathlib import Path

from .paths import JAIL_SOCK, JAIL_SRC, PROXY_PORT, PROXY_SOCK_NAME, PROXY_URL


class ClaudeContainerLaunch:
    def __init__(
        self,
        *,
        container: str,
        runtime,
        stage_src: str,
        proxy_sock_dir: str,
        workdir: str,
        proxy_host: "str | None" = None,
        network: "str | None" = None,
    ) -> None:
        self.container = container
        self.runtime = runtime
        self.stage_src = stage_src
        self.proxy_sock_dir = proxy_sock_dir
        self.proxy_host = proxy_host
        self.network = network
        self.workdir = str(workdir)

    @property
    def popen_env(self):
        return self.runtime.popen_env

    def wrap(self, base_argv: list[str], config_dir: "str | None", claude_env: dict) -> list[str]:
        config_dir = str(Path(config_dir or os.environ.get("CLAUDE_CONFIG_DIR")
                              or Path.home() / ".claude").expanduser().resolve())
        claude_cli = ["claude", *base_argv[1:]]

        docker = self.runtime.proxy_in_container
        proxy_url = f"http://{self.proxy_host}:{PROXY_PORT}" if docker else PROXY_URL

        jail_env = {
            "HOME": self.runtime.container_home,
            "PATH": "/usr/local/bin:/usr/local/sbin:/usr/bin:/usr/sbin:/bin:/sbin",
            "PYTHONPATH": JAIL_SRC,
            "PYTHONUNBUFFERED": "1",
            "HTTP_PROXY": proxy_url, "HTTPS_PROXY": proxy_url, "ALL_PROXY": proxy_url,
            "http_proxy": proxy_url, "https_proxy": proxy_url, "all_proxy": proxy_url,
            "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
            "DISABLE_AUTOUPDATER": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "IS_SANDBOX": "1",
        }
        jail_env.update(claude_env)
        jail_env["CLAUDE_CONFIG_DIR"] = config_dir

        mounts = [(self.stage_src, JAIL_SRC, "ro")]
        if not docker:
            mounts.append((self.proxy_sock_dir, JAIL_SOCK, "rw"))
        mounts.append((config_dir, config_dir, "rw"))

        probe_addr = f'("{self.proxy_host}",{PROXY_PORT})' if docker else f'("127.0.0.1",{PROXY_PORT})'
        bridge_up = "" if docker else (
            f"python3 {JAIL_SRC}/sandbox/jail_bridge.py {PROXY_PORT}:{JAIL_SOCK}/{PROXY_SOCK_NAME} "
            f">/tmp/claude_bridge.log 2>&1 &\n"
            "BR=$!\n"
            "trap 'kill $BR 2>/dev/null || true' EXIT INT TERM\n"
        )
        bridge_down = "" if docker else "kill $BR 2>/dev/null || true\n"
        inner_sh = (
            bridge_up
            + "for _ in $(seq 1 100); do "
            "python3 -c 'import socket,sys; "
            f"s=socket.create_connection({probe_addr},0.2); s.close()' 2>/dev/null "
            "&& break || sleep 0.1; done\n"
            f"mkdir -p {shlex.quote(self.workdir)} && cd {shlex.quote(self.workdir)}\n"
            f"{shlex.join(claude_cli)}\n"
            "rc=$?\n"
            + bridge_down
            + "exit $rc\n"
        )

        return self.runtime.container_cmd(
            name=self.container, mounts=mounts, env=jail_env,
            argv=["bash", "-c", inner_sh], network=self.network,
            tmpfs=[self.workdir] if self.runtime.needs_cwd_tmpfs else None)
