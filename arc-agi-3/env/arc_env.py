from __future__ import annotations

import os
from pathlib import Path

from arc_agi import Arcade, OperationMode

_DEFAULT_ENV_DIR = Path(__file__).resolve().parent / "assets"


class ArcEnv:

    def __init__(self, game_id: str) -> None:
        self.mode = "offline"
        os.environ["ONLY_RESET_LEVELS"] = "true"
        self.arc = Arcade(
            arc_api_key=os.environ.get("ARC_API_KEY", ""),
            arc_base_url="https://three.arcprize.org",
            operation_mode=OperationMode.OFFLINE,
            environments_dir=str(_DEFAULT_ENV_DIR),
        )
        self.game_id = self._resolve_game_id(game_id)
        self._make_kwargs = dict(seed=0, scorecard_id=None, render_mode=None, include_frame_data=True)
        self._wrapper = self._make_wrapper()

    def _make_wrapper(self):
        w = self.arc.make(self.game_id, **self._make_kwargs)
        if w is None:
            raise RuntimeError(f"cannot create environment game_id={self.game_id!r}")
        return w

    def _resolve_game_id(self, game_id: str) -> str:
        envs = {e.game_id: e for e in self.arc.get_environments()}
        if game_id in envs:
            return game_id
        cands = [gid for gid in envs if gid.split("-", 1)[0] == game_id]
        if len(cands) == 1:
            return cands[0]
        if not envs:
            return game_id
        if not cands:
            raise ValueError(f"unknown game {game_id!r}; available: {sorted(envs)}")
        raise ValueError(f"game {game_id!r} matches several versions: {cands}")

    @property
    def wrapper(self):
        return self._wrapper
