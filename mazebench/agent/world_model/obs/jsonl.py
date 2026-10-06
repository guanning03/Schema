from __future__ import annotations

import json
import threading
from pathlib import Path

from ..events import Event, EventSink


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

    def close(self) -> None:
        with self._lock:
            try:
                self._fh.close()
            except Exception:
                pass
