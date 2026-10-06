"""The world model never sees the absolute step number: MazeBench physics does not depend on it,
and a model that can read it can replace modelling with `if action_count == N` lookups."""
from __future__ import annotations

from env.maze_env import MazeState
from agent.world_model.timestep import meta_of
from agent.world_model.tools import build_system_prompt, write_current_board
from agent.world_model.world import CodeWorldModel, advance_meta

SEEN = '''
SEEN = []
def init_state(entry_obs, meta, prev_state=None):
    SEEN.append(dict(meta)); return {}
def predict(state, obs, action, meta):
    SEEN.append(dict(meta)); return obs, {}, state
'''


def test_meta_has_no_step_number():
    st = MazeState(room="level_HxI", view="top", yaw=0, gems=3, visited_rooms=["level_HxI"],
                   action_count=12345, observation="P")
    m = meta_of(st)
    assert "action_count" not in m
    assert set(m) == {"room", "view", "yaw", "gems", "visited_rooms"}
    assert "action_count" not in advance_meta(m, "up", {})


def test_model_calls_never_receive_it():
    w = CodeWorldModel(SEEN)
    m = meta_of(MazeState(room="level_HxI", view="top", yaw=0, action_count=777, observation="P"))
    w.set_entry({"obs": "P", "meta": m})
    w.init_state("P", m, None)
    w.predict({}, "P", "up", advance_meta(m, "up", {}))
    assert all("action_count" not in s for s in w._ns["SEEN"])
    assert "action_count" not in (w._ns["ENTRY_META"] or {})


def test_current_board_and_prompt(tmp_path):
    m = meta_of(MazeState(room="level_HxI", view="top", yaw=0, action_count=42, observation="P"))
    write_current_board(tmp_path, "P", m, "v")
    assert "action_count" not in (tmp_path / "current_board.py").read_text()
    assert "action_count" not in build_system_prompt(1000)
