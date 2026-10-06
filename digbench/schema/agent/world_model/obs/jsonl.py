from __future__ import annotations

import json
import threading
from pathlib import Path

from ..events import Event, EventSink, RunStarted


class JsonlSink(EventSink):
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fh = self.path.open("a", encoding="utf-8")

    def emit(self, event: Event) -> None:
        line = json.dumps(event.to_dict(), ensure_ascii=False)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()
        if isinstance(event, RunStarted):
            self._write_run_json(event)

    def _write_run_json(self, ev: RunStarted) -> None:
        meta = {
            "game_id": ev.game_id,
            "session_id": ev.session_id,
            "base_url": ev.base_url,
            "provider": ev.provider,
            "model": ev.model,
            "model_name": ev.model_name,
            "max_actions": ev.max_actions,
            "max_level": ev.max_level,
            "description": ev.description,
            "seed": ev.seed,
            "framework_version": ev.framework_version,
            "workdir": ev.workdir,
            "started_at": ev.ts,
        }
        (self.path.parent / "run.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def close(self) -> None:
        with self._lock:
            try:
                self._fh.close()
            except Exception:
                pass
