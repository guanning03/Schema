from __future__ import annotations

from .clientutil import parse_overflow_tokens
from .types import Policy

MIN_KEEP_TURNS = 2

MAX_OVERFLOW_ROUNDS = 8
_OVERFLOW_MARGIN = 1.25
_OVERFLOW_SLACK = 1000


def truncate_if_needed(policy: Policy, budget: int | None, *, log=None) -> int:
    if not budget or budget <= 0:
        return 0
    last = policy.last_prompt_tokens
    if last <= 0:
        return 0
    if last <= budget:
        return 0

    projected = last
    evicted = 0
    while projected > budget:
        turns = policy.turns()
        if len(turns) <= MIN_KEEP_TURNS:
            break
        oldest = turns[0]
        if oldest.is_active:
            break
        before = len(turns)
        policy.evict_oldest_turn()
        if len(policy.turns()) >= before:
            break
        projected -= max(0, oldest.est_tokens)
        evicted += 1

    if evicted and log:
        kept = len(policy.turns())
        if projected <= budget:
            log(
                f"context truncated: evicted {evicted} step-pair(s), projected prompt "
                f"~{projected} <= budget {budget}, kept {kept} unit(s)"
            )
        else:
            log(
                f"context truncated: evicted {evicted} step-pair(s), projected prompt "
                f"~{projected} > budget {budget} (floor MIN_KEEP_TURNS={MIN_KEEP_TURNS} "
                f"reached, still over budget), kept {kept} unit(s)"
            )
    return evicted


def evict_for_overflow(policy: Policy, err, *, round_idx: int = 0) -> int:
    parsed = parse_overflow_tokens(err)
    target = None
    if parsed:
        total, limit = parsed
        target = int((total - limit) * _OVERFLOW_MARGIN) + _OVERFLOW_SLACK
    evicted = 0
    removed_est = 0
    while (removed_est < target) if target is not None else (evicted < 2 ** round_idx):
        turns = policy.turns()
        if not turns or turns[0].is_active:
            break
        before = len(turns)
        policy.evict_oldest_turn()
        if len(policy.turns()) >= before:
            break
        removed_est += max(0, turns[0].est_tokens)
        evicted += 1
    return evicted
