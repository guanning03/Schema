from __future__ import annotations

import time
from typing import Optional

_SENT_AT: list = [None]


def set_sent_at(t: Optional[float]) -> None:
    _SENT_AT[0] = t


def sent_at() -> Optional[float]:
    return _SENT_AT[0]


def queued_s() -> float:
    t = _SENT_AT[0]
    return max(0.0, time.monotonic() - t) if t is not None else 0.0
