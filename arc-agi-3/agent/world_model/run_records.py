from __future__ import annotations

from pathlib import Path

MODEL_FILE = "world_model.py"

RECORD_FILES: "frozenset[str]" = frozenset({"events.jsonl", "run.json"})
RECORD_DIRS: "frozenset[str]" = frozenset({"sessions", "session_live", "snapshots", ".git"})

RECORD_NAMES: "tuple[str, ...]" = (
    "events.jsonl", "run.json", "sessions", "session_live", "snapshots", ".git",
)


def is_record(workdir: "str | Path", path: "str | Path") -> bool:
    try:
        rel = Path(path).resolve().relative_to(Path(workdir).resolve())
    except (ValueError, OSError):
        return False
    parts = rel.parts
    if not parts:
        return False
    top = parts[0]
    return (len(parts) == 1 and top in RECORD_FILES) or top in RECORD_DIRS


def record_paths(root: "str | Path") -> "list[str]":
    base = str(root).rstrip("/")
    return [f"{base}/{name}" for name in RECORD_NAMES]
