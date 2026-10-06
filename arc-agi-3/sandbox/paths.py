from __future__ import annotations

from pathlib import Path

JAIL_SRC = "/opt/arc/src"
JAIL_SOCK = "/opt/arc/sock"

PROXY_PORT = 3128
PROXY_URL = f"http://127.0.0.1:{PROXY_PORT}"
PROXY_SOCK_NAME = "proxy.sock"

STAGED_TOPS = ("agent", "sandbox")


def jail_to_host(jail_path: "str | Path", repo_root: "str | Path") -> "Path | None":
    p = Path(jail_path)
    try:
        rel = p.relative_to(JAIL_SRC)
    except ValueError:
        return None
    if not rel.parts or rel.parts[0] not in STAGED_TOPS:
        return None
    return Path(repo_root).resolve() / rel
