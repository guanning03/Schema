"""Carried state across segments: init_state(entry_obs, meta, prev_state).

A toy game with one invisible, persistent fact: H = how many times `x` was pressed, ever. Pressing
`x` shows `*` when H is even and `#` when odd, so predicting it needs H from earlier segments.
`go` changes room and ends the segment. A two-argument model restarts H at 0 every segment and
mispredicts; a carrying model gets it right — and the harness must chain the starts exactly the
same way whether it walks serially or verifies guesses in parallel.
"""
from __future__ import annotations

import random

import pytest

from agent.world_model.timestep import segment_entries
from agent.world_model.world import (_BT_LAST, CodeWorldModel, SegmentStarts, _UNSET,
                                        backtest_rollout, chain_segments, rollout_state,
                                        tried_actions)

ROOMS = "ABCD"


class FakeTS:
    def __init__(self, action, before_obs, after, segment, before_meta, after_meta, i):
        self.action, self.before_obs, self.after, self.segment = action, before_obs, after, segment
        self.before_meta, self.after_meta = before_meta, after_meta
        self.invalid = False
        self.room_changed = action == "go"
        self.gem = self.dead = self.won = False
        self.env_step_index = i + 1

    @property
    def terminal(self):
        return self.room_changed


def make_timeline(n=900, seed=0):
    rnd = random.Random(seed)
    room, local, H, seg = 0, 0, 0, 0
    obs = f"{ROOMS[room]}{local}"
    tl = []
    for i in range(n):
        a = rnd.choices(["a", "x", "go"], weights=[6, 3, 1])[0]
        bm = {"room": f"level_{ROOMS[room]}xA", "view": "top", "yaw": 0, "gems": 0,
              "visited_rooms": [], "action_count": i}
        if a == "go":
            room, local = (room + 1) % len(ROOMS), 0
            after = f"{ROOMS[room]}{local}"
        elif a == "x":
            local += 1
            after = f"{ROOMS[room]}{local}{'*' if H % 2 == 0 else '#'}"
            H += 1
        else:
            local += 1
            after = f"{ROOMS[room]}{local}"
        am = dict(bm, room=f"level_{ROOMS[room]}xA", action_count=i + 1)
        tl.append(FakeTS(a, obs, after, seg, bm, am, i))
        obs = after
        if a == "go":
            seg += 1
    return tl


PREDICT = '''
def predict(state, obs, action, meta):
    s = dict(state)
    room = obs[0]
    if action == "go":
        room = "ABCD"[("ABCD".index(room) + 1) % 4]
        s["local"] = 0
        return f"{room}{s['local']}", {"room_changed": True}, s
    s["local"] += 1
    if action == "x":
        mark = "*" if s["H"] % 2 == 0 else "#"
        s["H"] += 1
        return f"{room}{s['local']}{mark}", {}, s
    return f"{room}{s['local']}", {}, s
'''
TWO_ARG = PREDICT + '''
def init_state(entry_obs, meta):
    return {"H": 0, "local": 0}
'''
CARRY = PREDICT + '''
def init_state(entry_obs, meta, prev_state=None):
    return {"H": prev_state["H"] if prev_state else 0, "local": 0}
'''
# same carry, different code (a new model version): every guess from CARRY stays right
CARRY_EDITED = CARRY + "\n# an edit that does not change what is carried\n"
# wrong for room C only: entering C forgets H (guesses go wrong from the first C onwards,
# and the chain re-agrees with them later only where H's parity happens to match)
CARRY_FORGETS_IN_C = PREDICT + '''
def init_state(entry_obs, meta, prev_state=None):
    keep = prev_state["H"] if prev_state and entry_obs[0] != "C" else 0
    return {"H": keep, "local": 0}
'''


def run_bt(code, tl, starts=None, workers=1, segments=None):
    w = CodeWorldModel(code)
    res = backtest_rollout(w, tl, segment_entries(tl), segments=segments, workers=workers,
                           starts=starts)
    return {i: (r["errors"], r["pred"]) for i, r in res.items()}, dict(_BT_LAST)


def n_bad(res):
    return sum(1 for errs, _ in res.values() if errs)


@pytest.fixture(scope="module")
def tl():
    return make_timeline()


def test_two_arg_model_is_untouched(tl):
    w = CodeWorldModel(TWO_ARG)
    assert not w.carries
    plain, _ = run_bt(TWO_ARG, tl)
    with_cache, _ = run_bt(TWO_ARG, tl, starts=SegmentStarts())
    assert plain == with_cache                           # the cache is ignored for non-carriers
    assert n_bad(plain) > 0                             # and the toy really needs carried state


def test_carrying_model_reproduces_history(tl):
    w = CodeWorldModel(CARRY)
    assert w.carries
    res, stats = run_bt(CARRY, tl, starts=SegmentStarts())
    assert len(res) == len(tl) and n_bad(res) == 0
    assert stats["mode"].startswith("chain")


def test_parallel_verification_equals_serial(tl):
    cache = SegmentStarts()
    serial, _ = run_bt(CARRY, tl, starts=cache, workers=1)         # version 1: no guesses
    par, stats = run_bt(CARRY_EDITED, tl, starts=cache, workers=4)  # version 2: v1's starts guess
    assert par == serial
    assert stats["mode"] == "chain-parallel", stats
    assert stats["serial"] == 0 and stats["confirmed"] > 0, stats


def test_wrong_guesses_rerun_only_where_needed(tl):
    cache = SegmentStarts()
    run_bt(CARRY, tl, starts=cache, workers=1)
    par, stats = run_bt(CARRY_FORGETS_IN_C, tl, starts=cache, workers=4)
    ref, _ = run_bt(CARRY_FORGETS_IN_C, tl, starts=SegmentStarts(), workers=1)
    assert par == ref                                   # exact, despite partly wrong guesses
    n_segments = tl[-1].segment + 1
    assert 0 < stats["serial"] < n_segments, stats      # some re-run serially, not all
    assert stats["confirmed"] > 0, stats


def test_filtered_backtest_and_cache_reuse(tl):
    cache = SegmentStarts()
    full, _ = run_bt(CARRY, tl, starts=cache)
    last_seg = tl[-1].segment
    part, stats = run_bt(CARRY, tl, starts=cache, segments={last_seg})
    assert part == {i: v for i, v in full.items() if tl[i].segment == last_seg}
    assert stats["starts_computed"] == 0                # every start was already cached


def test_rollout_from_carried_start(tl):
    w = CodeWorldModel(CARRY)
    cache = SegmentStarts()
    entries = segment_entries(tl)
    seg = tl[-1].segment
    chain_segments(w, tl, entries, cache, seg)
    state, meta, errs = rollout_state(w, tl, entries, seg, len(tl), start_state=cache.get(seg))
    assert not errs
    H = sum(1 for ts in tl if ts.action == "x")
    assert state["H"] == H


def test_init_state_error_is_reported_and_chain_goes_on(tl):
    bad = PREDICT + '''
def init_state(entry_obs, meta, prev_state=None):
    if entry_obs[0] == "B" and meta.get("action_count", 0) > 300:
        raise RuntimeError("boom")
    return {"H": prev_state["H"] if prev_state else 0, "local": 0}
'''
    res, _ = run_bt(bad, tl, starts=SegmentStarts())
    boom = [i for i, (errs, _) in res.items() if any("init_state raised" in e for e in errs)]
    assert boom                                         # reported at each failing segment start
    assert any(not errs for i, (errs, _) in res.items() if i > max(boom))   # chain continues


def test_tried_actions_uses_cached_starts_only(tl):
    w = CodeWorldModel(CARRY)
    entries = segment_entries(tl)
    cache = SegmentStarts()
    _, complete = tried_actions(w, tl, entries, None, {"level_AxA"}, {"x"},
                                start_of=lambda s: cache.cached(w, s))
    assert complete is False                            # nothing cached yet: skipped, not guessed
    chain_segments(w, tl, entries, cache, tl[-1].segment)
    out, complete = tried_actions(w, tl, entries, None, {"level_AxA"}, {"x"},
                                  start_of=lambda s: cache.cached(w, s))
    assert complete is True and out
    assert cache.cached(CodeWorldModel(CARRY_EDITED), 0) is _UNSET   # other version: unknown
