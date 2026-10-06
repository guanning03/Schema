from __future__ import annotations


def match_model(model: str, table: dict):
    best = None
    for key, value in table.items():
        if key in model and (best is None or len(key) > len(best[0])):
            best = (key, value)
    return best[1] if best else None


def pricing_for(model: str, table: dict) -> dict | None:
    return match_model(model, table)


def compute_cost(
    model: str, prompt: int, cached: int, output: int, thoughts: int, table: dict
) -> float | None:
    pricing = pricing_for(model, table)
    if pricing is None:
        return None
    tier = pricing.get("long_context")
    if tier and prompt > tier["threshold"]:
        pricing = {**pricing, **{k: v for k, v in tier.items() if k != "threshold"}}
    uncached = max(0, prompt - cached)
    cost = (
        uncached * pricing["input_per_1m"]
        + cached * pricing.get("cached_input_per_1m", pricing["input_per_1m"])
        + output * pricing["output_per_1m"]
        + thoughts * pricing.get("thoughts_per_1m", pricing["output_per_1m"])
    ) / 1_000_000
    return round(cost, 8)
