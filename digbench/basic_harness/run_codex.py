from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from clients.codex_client import CodexPolicy
from core.bench import Bench
from core.harness import play
from core.output import Output

SERVER = "https://api.digbench.ai"
BASE = SERVER + "/api/agent"
BENCH_TIMEOUT = 60

_BUDGET_SLACK = 2000
_OUTPUT_RESERVE_FACTOR = 1.5


def game_token() -> str:
    return os.environ.get("DIGBENCH_API_TOKEN", "").strip()


def resolve_budget(args, model_max_context: int | None, *, warn=None) -> int | None:
    if args.context_budget and args.context_budget > 0:
        return args.context_budget
    if model_max_context:
        reserve = max(int(_OUTPUT_RESERVE_FACTOR * args.max_tokens) + _BUDGET_SLACK,
                      int((1 - args.context_proportion) * model_max_context))
        budget = model_max_context - reserve
        if budget <= 0:
            if warn:
                warn(f"--max-tokens ({args.max_tokens}) + reserve exceeds the model window "
                     f"({model_max_context}); lower --max-tokens or set --context-budget. "
                     "Proactive truncation off (the overflow evict-retry still bounds context, "
                     "less efficiently).")
            return None
        return budget
    if warn:
        warn(
            f"no context-window known for {args.model!r} and no --context-budget set: "
            "context truncation DISABLED (append-only). Pass --context-budget to enable it."
        )
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("game")
    p.add_argument("--model", default="gpt-6-astra")
    p.add_argument("--effort", default="max")
    p.add_argument("--run-dir", default="runs")
    a = p.parse_args()
    a.max_tokens = 32000
    a.max_steps = 3000
    a.max_invalid_retries = 5
    a.max_cost_usd = 0.0
    a.max_api_retries = 10
    a.api_timeout_seconds = 600
    a.context_proportion = 1.0
    a.context_budget = 0
    a.summary_chars = 300
    a.verbose = False
    token = game_token()
    if not token: sys.exit("no DIGBENCH_API_TOKEN")
    bench = Bench(BASE, token, timeout=BENCH_TIMEOUT, max_retries=a.max_api_retries)
    run_dir = Path(a.run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    policy = CodexPolicy(model=a.model, effort=a.effort, max_tokens=a.max_tokens,
                         timeout=a.api_timeout_seconds, max_retries=a.max_api_retries)
    policy.debug_dir = str(run_dir)
    a.model = policy.model
    run_label = f"basic-harness_{a.model}-{a.effort}"
    budget = resolve_budget(a, policy.model_max_context, warn=lambda m: print("  ⚠️ ", m))
    print(f"[basic-harness] {a.game} with codex:{a.model} effort={a.effort} budget={budget or 'off'}")
    start = bench.start_session(a.game, run_label, a.model)
    sid = start["session_id"]
    base_name = f"{run_label}_{time.strftime('%Y%m%d-%H%M%S')}_{sid[:8]}".replace("/", "_")
    out = Output(summary_chars=a.summary_chars, verbose=a.verbose,
                 log_path=run_dir/(base_name+".log"), jsonl_path=run_dir/(base_name+".jsonl"))
    try:
        play(bench, policy, out, start, a, SERVER, a.effort, run_label, context_budget=budget, provenance=None)
    finally:
        out.close()
    print(f"saved: {run_dir/base_name}.log")

if __name__ == "__main__":
    main()
