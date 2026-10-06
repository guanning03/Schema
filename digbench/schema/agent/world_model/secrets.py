from __future__ import annotations

import os

SECRET_ENV_PREFIXES: tuple[str, ...] = ("DIGBENCH_",)
SECRET_ENV_NAMES: frozenset[str] = frozenset({"DIGBENCH_API_TOKEN"})


def is_secret_var(name: str) -> bool:
    return name in SECRET_ENV_NAMES or any(name.startswith(p) for p in SECRET_ENV_PREFIXES)


def scrub_env() -> dict:
    return {k: v for k, v in os.environ.items() if not is_secret_var(k)}
