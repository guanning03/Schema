from __future__ import annotations

import json
import time
from typing import Any

from core import accounting, clientutil, prompts
from core.types import Move, Turn

OPENAI_PRICING: dict[str, dict[str, float]] = {
    "gpt-5.5":  {"input_per_1m": 5.0,  "cached_input_per_1m": 0.50, "output_per_1m": 30.0, "thoughts_per_1m": 30.0,
                 "long_context": {"threshold": 272_000,
                                  "input_per_1m": 10.0, "cached_input_per_1m": 1.0,
                                  "output_per_1m": 45.0, "thoughts_per_1m": 45.0}},
    "gpt-5.4-mini": {"input_per_1m": 0.75, "cached_input_per_1m": 0.075, "output_per_1m": 4.50, "thoughts_per_1m": 4.50},
    "gpt-5.4":  {"input_per_1m": 2.50, "cached_input_per_1m": 0.25, "output_per_1m": 15.0, "thoughts_per_1m": 15.0},
    "o3":       {"input_per_1m": 2.0,  "cached_input_per_1m": 0.50, "output_per_1m": 8.0,  "thoughts_per_1m": 8.0},
    "o4-mini":  {"input_per_1m": 1.10, "cached_input_per_1m": 0.275, "output_per_1m": 4.40, "thoughts_per_1m": 4.40},
}

OPENAI_MAX_CONTEXT: dict[str, int] = {
    "openai.gpt-5.5":  272_000,
    "openai.gpt-5.4":  272_000,
    "gpt-5.5":       1_050_000,
    "gpt-5.4-mini":    400_000,
    "gpt-5.4":       1_050_000,
    "gpt-4.1":       1_000_000,
    "o3":              200_000,
    "o4-mini":         200_000,
}

OPENAI_LONG_CONTEXT_THRESHOLD = 272_000


def _make_move_tool() -> dict:
    s = prompts.MAKE_MOVE_SPEC
    return {"type": "function", "name": s["name"], "description": s["description"], "parameters": s["parameters"]}


def _item_type(item) -> str | None:
    if isinstance(item, dict):
        return item.get("type") or ("message" if "role" in item else None)
    return getattr(item, "type", None)


class OpenAIPolicy:

    def __init__(
        self,
        *,
        model: str,
        effort: str,
        max_tokens: int,
        timeout: int,
        max_retries: int,
        pricing: dict | None = None,
        include_thoughts: bool = True,
        base_url: str | None = None,
        client,
    ):
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.debrief_retries = max(1, min(2, max_retries))
        self.pricing = pricing or OPENAI_PRICING
        self.include_thoughts = include_thoughts
        self.move_channel = "forced-tool"
        self.thoughts_basis = "exact"

        self.pricing_row = accounting.pricing_for(model, self.pricing)
        self.has_pricing = self.pricing_row is not None
        self.model_max_context = accounting.match_model(model, OPENAI_MAX_CONTEXT)

        self.client = client

        self.input: list = []
        self._last_call_id: str | None = None
        self.call_count = 0
        self.prompt_tokens = 0
        self.cached_tokens = 0
        self.output_tokens = 0
        self.thoughts_tokens = 0
        self.total_tokens = 0
        self.cost_usd = 0.0
        self.elapsed_s = 0.0
        self.last_prompt_tokens = 0
        self.base_url = base_url
        self.reported_model: str | None = None
        self._warned_long_context = False
        self._last_error: dict | None = None
        self.on_retry = None
        self.debug_dir = "."


    def start(self, description: str, state_slice: dict) -> None:
        self.input.append({"role": "user", "content": prompts.seed_text(description, state_slice)})

    def observe(self, result: dict) -> None:
        payload = json.dumps({"result": result})
        if self._last_call_id is not None:
            self.input.append({"type": "function_call_output", "call_id": self._last_call_id, "output": payload})
            self._last_call_id = None
        else:
            self.input.append({"role": "user", "content": payload})

    def add_nudge(self, text: str) -> None:
        if self._last_call_id is not None:
            self.input.append({"type": "function_call_output", "call_id": self._last_call_id, "output": text})
            self._last_call_id = None
        else:
            self.input.append({"role": "user", "content": text})


    def _reasoning_param(self) -> dict:
        r: dict[str, Any] = {"effort": self.effort}
        if self.effort != "none" and self.include_thoughts:
            r["summary"] = "auto"
        return r

    def _turn_kwargs(self) -> dict:
        return {
            "model": self.model,
            "instructions": prompts.SYSTEM_INSTRUCTION,
            "input": self.input,
            "tools": [_make_move_tool()],
            "tool_choice": {"type": "function", "name": "make_move"},
            "parallel_tool_calls": False,
            "reasoning": self._reasoning_param(),
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "max_output_tokens": self.max_tokens,
        }

    def _debrief_kwargs(self) -> dict:
        return {
            "model": self.model,
            "instructions": prompts.DEBRIEF_SYSTEM_INSTRUCTION,
            "input": self.input,
            "reasoning": self._reasoning_param(),
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "max_output_tokens": self.max_tokens,
        }


    def _create(self, build_kwargs, max_retries: int, timeout: int | None = None):
        kwargs = build_kwargs()
        opts = {} if timeout is None else {"timeout": timeout}
        resp, err = clientutil.run_request(
            lambda: self.client.responses.create(**kwargs, **opts),
            provider="OpenAI", max_retries=max_retries, request=kwargs,
            debug_dir=self.debug_dir, call_count=self.call_count, on_event=self.on_retry,
        )
        self._last_error = err
        return resp

    @staticmethod
    def _extract(resp) -> tuple[str | None, str | None, str, bool, str | None]:
        action = call_id = None
        summary_parts, has_enc = [], False
        for item in getattr(resp, "output", None) or []:
            itype = getattr(item, "type", None)
            if itype == "function_call" and getattr(item, "name", None) == "make_move":
                try:
                    args = json.loads(getattr(item, "arguments", "") or "{}")
                except (TypeError, ValueError):
                    args = {}
                if isinstance(args, dict):
                    action = str(args.get("action") or "").strip() or None
                call_id = getattr(item, "call_id", None)
            elif itype == "reasoning":
                if getattr(item, "encrypted_content", None):
                    has_enc = True
                for s in getattr(item, "summary", None) or []:
                    text = getattr(s, "text", None)
                    if text:
                        summary_parts.append(text.strip())
        return action, call_id, " ".join(p for p in summary_parts if p), has_enc, getattr(resp, "status", None)

    def _account(self, usage) -> tuple[dict, float | None]:
        inp = int(getattr(usage, "input_tokens", 0) or 0)
        out_total = int(getattr(usage, "output_tokens", 0) or 0)
        odetails = getattr(usage, "output_tokens_details", None)
        reasoning = int(getattr(odetails, "reasoning_tokens", 0) or 0)
        idetails = getattr(usage, "input_tokens_details", None)
        cached = int(getattr(idetails, "cached_tokens", 0) or 0)
        visible = max(0, out_total - reasoning)
        counts = {"prompt": inp, "cached": cached, "output": visible, "thoughts": reasoning,
                  "total": inp + out_total}
        cost = accounting.compute_cost(self.model, inp, cached, visible, reasoning, self.pricing)
        self.call_count += 1
        self.prompt_tokens += inp
        self.cached_tokens += cached
        self.output_tokens += visible
        self.thoughts_tokens += reasoning
        self.total_tokens += counts["total"]
        if inp:
            self.last_prompt_tokens = inp
        if cost is not None:
            self.cost_usd += cost
        if inp > OPENAI_LONG_CONTEXT_THRESHOLD and not self._warned_long_context:
            self._warned_long_context = True
            if self.on_retry:
                if self.pricing_row and self.pricing_row.get("long_context"):
                    self.on_retry(f"prompt {inp} > {OPENAI_LONG_CONTEXT_THRESHOLD}: "
                                  "long-context tier rates now apply (per call, priced)")
                else:
                    self.on_retry(f"prompt {inp} > {OPENAI_LONG_CONTEXT_THRESHOLD}: cost may be "
                                  "underestimated (no long-context tier priced for this model)")
        return counts, cost

    def generate_move(self) -> Move:
        start = time.time()
        resp = self._create(self._turn_kwargs, self.max_retries)
        elapsed = time.time() - start
        self.elapsed_s += elapsed
        if resp is None:
            return Move(None, "", False, None, clientutil.zero_usage(), None, elapsed, error=self._last_error)

        if self.reported_model is None:
            self.reported_model = getattr(resp, "model", None)
        self.input += list(getattr(resp, "output", None) or [])
        action, call_id, summary, has_enc, status = self._extract(resp)
        self._last_call_id = call_id
        if call_id is None:
            while self.input and _item_type(self.input[-1]) == "reasoning":
                self.input.pop()
        counts, cost = self._account(getattr(resp, "usage", None))
        return Move(action, summary, has_enc, status, counts, cost, elapsed,
                    thoughts_basis=self.thoughts_basis,
                    continuity="verified" if has_enc else None)

    def debrief(self) -> str | None:
        self.add_nudge(prompts.DEBRIEF_PROMPT)
        start = time.time()
        resp = self._create(self._debrief_kwargs, self.debrief_retries, timeout=2 * self.timeout)
        self.elapsed_s += time.time() - start
        if resp is None:
            return None
        self._account(getattr(resp, "usage", None))
        text = (getattr(resp, "output_text", None) or "").strip()
        return text or "(empty debrief)"


    @staticmethod
    def _is_model_item(item) -> bool:
        t = _item_type(item)
        if t in ("reasoning", "function_call"):
            return True
        if t == "message":
            role = item.get("role") if isinstance(item, dict) else getattr(item, "role", None)
            return role == "assistant"
        return False

    def _turn_start_indices(self) -> list[int]:
        out = []
        for i, it in enumerate(self.input):
            if self._is_model_item(it) and (i == 0 or not self._is_model_item(self.input[i - 1])):
                out.append(i)
        return out

    def turns(self):
        starts = self._turn_start_indices()
        if not starts:
            return []
        bounds = starts + [len(self.input)]
        n = len(starts)
        out = []
        for k in range(n):
            chunk = self.input[bounds[k]:bounds[k + 1]]
            out.append(Turn(payload=chunk, is_active=(k == n - 1), est_tokens=clientutil.est_tokens(chunk)))
        return out

    def evict_oldest_turn(self) -> None:
        starts = self._turn_start_indices()
        if len(starts) <= 1:
            return
        del self.input[starts[0]:starts[1]]
