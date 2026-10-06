from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.request
from typing import Callable

_PERMANENT_429_MARKERS = ("insufficient_quota",)

BACKOFF_CAP = 60

THINKING_LEVEL_BUDGETS: dict[str, int] = {
    "minimal": 1024, "low": 2048, "medium": 8192,
    "high": 16384, "xhigh": 24576, "max": 32768,
}


def clamp_thinking_budget(level: str, max_tokens: int, *, floor: int, cap: int | None = None) -> int:
    budget = min(THINKING_LEVEL_BUDGETS[level], max_tokens - 1024)
    if cap is not None:
        budget = min(budget, cap)
    if budget < max(floor, 1):
        raise ValueError(
            f"--max-tokens {max_tokens} leaves no room for a thinking budget at level "
            f"{level!r} (needs >= {max(floor, 1) + 1024} total): raise --max-tokens or use "
            "--thinking-level none"
        )
    return budget


def backoff_sleep(attempt: int, cap: int = BACKOFF_CAP) -> None:
    time.sleep(min(2 ** attempt, cap) + random.uniform(0, 1 + attempt))


class _RefuseRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_no_redirect_opener = urllib.request.build_opener(_RefuseRedirect)


def urlopen_no_redirect(req, timeout):
    return _no_redirect_opener.open(req, timeout=timeout)


class StreamError(RuntimeError):

    def __init__(self, code, body):
        super().__init__(str(body))
        self.code = code
        self.body = body


def iter_sse_data(line_iterable):
    for raw in line_iterable:
        line = (raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else raw).strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data:
            continue
        if data == "[DONE]":
            return
        yield data


def accumulate_chat_stream(line_iterable, *, require_usage: bool = False) -> dict:
    content: list = []
    reasoning: list = []
    reasoning_key = None
    tool_calls: dict = {}
    finish_reason = None
    usage = None
    model = None
    for data in iter_sse_data(line_iterable):
        chunk = json.loads(data)
        if chunk.get("error"):
            err = chunk["error"]
            code = err.get("code") if isinstance(err, dict) else None
            raise StreamError(code if isinstance(code, int) else None, err)
        if chunk.get("model"):
            model = chunk["model"]
        if chunk.get("usage"):
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content.append(delta["content"])
            for key in ("reasoning_content", "reasoning"):
                if delta.get(key):
                    reasoning.append(delta[key])
                    reasoning_key = reasoning_key or key
                    break
            for tc in delta.get("tool_calls") or []:
                slot = tool_calls.setdefault(
                    tc.get("index", 0),
                    {"id": None, "type": "function", "function": {"name": None, "arguments": ""}},
                )
                if tc.get("id"):
                    slot["id"] = tc["id"]
                if tc.get("type"):
                    slot["type"] = tc["type"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["function"]["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["function"]["arguments"] += fn["arguments"]
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
    if finish_reason is None:
        raise StreamError(None, "truncated stream (no finish_reason)")
    if require_usage and usage is None:
        raise StreamError(None, "truncated stream (finish_reason without requested usage)")
    message: dict = {"role": "assistant", "content": "".join(content)}
    if reasoning_key is not None:
        message[reasoning_key] = "".join(reasoning)
    if tool_calls:
        message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
    result: dict = {"choices": [{"message": message, "finish_reason": finish_reason}]}
    if usage is not None:
        result["usage"] = usage
    if model is not None:
        result["model"] = model
    return result


def zero_usage() -> dict:
    return {"prompt": 0, "cached": 0, "output": 0, "thoughts": 0, "total": 0}


def est_tokens(chunk) -> int:
    return sum(len(str(c)) for c in chunk) // 4


def extract_status(exc) -> int | None:
    for attr in ("status_code", "code"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
    return None


def _is_permanent_429(exc) -> bool:
    blob = (str(getattr(exc, "code", "")) + " " + str(getattr(exc, "body", "")) + " " + str(exc)).lower()
    return any(m in blob for m in _PERMANENT_429_MARKERS)


def is_retryable(exc) -> bool:
    if isinstance(exc, (TypeError, AttributeError, KeyError, IndexError, NameError)):
        return False
    status = extract_status(exc)
    if status is None:
        return True
    if status == 429:
        return not _is_permanent_429(exc)
    if status == 408:
        return True
    if 300 <= status < 400:
        return False
    if 400 <= status < 500:
        return False
    return True


_CONTEXT_OVERFLOW_MARKERS = (
    "context_length_exceeded", "maximum context length", "exceed context limit",
    "context window", "reduce the length", "longer than the maximum",
    "exceeds the maximum number of tokens allowed",
)


def is_context_overflow(err) -> bool:
    if not err:
        return False
    blob = (str(err.get("message", "")) if isinstance(err, dict) else str(err)).lower()
    return any(m in blob for m in _CONTEXT_OVERFLOW_MARKERS)


_OVERFLOW_LIMIT_RE = re.compile(r"maximum context length (?:is |of )?(\d+)")
_OVERFLOW_TOTAL_RE = re.compile(r"requested (?:a total of )?(\d+) tokens")


def parse_overflow_tokens(err) -> tuple[int, int] | None:
    if not err:
        return None
    blob = (str(err.get("message", "")) if isinstance(err, dict) else str(err)).lower()
    limit_m = _OVERFLOW_LIMIT_RE.search(blob)
    total_m = _OVERFLOW_TOTAL_RE.search(blob)
    if not (limit_m and total_m):
        return None
    limit, total = int(limit_m.group(1)), int(total_m.group(1))
    if not (0 < limit < total):
        return None
    return total, limit


def _error_dict(exc, attempt: int, max_retries: int, dump: str | None = None) -> dict:
    err = {"type": type(exc).__name__, "message": str(exc),
           "attempt": attempt + 1, "max_retries": max_retries}
    if dump:
        err["payload_dump"] = dump
    return err


def dump_request(debug_dir: str, provider: str, exc, request, call_count: int) -> str | None:
    fname = os.path.join(debug_dir, f"{provider.lower()}_4xx_{time.strftime('%Y%m%d-%H%M%S')}_{call_count}.json")
    try:
        req = json.loads(json.dumps(request))
    except (TypeError, ValueError):
        req = {"request_keys": sorted(request.keys())} if isinstance(request, dict) else {"request_type": str(type(request))}
    try:
        with open(fname, "w", encoding="utf-8") as f:
            json.dump({"http_status": extract_status(exc), "error": str(exc), "request": req},
                      f, indent=2, ensure_ascii=False)
        return fname
    except Exception:
        return None


def run_request(
    call: Callable,
    *,
    provider: str,
    max_retries: int,
    request,
    debug_dir: str,
    call_count: int = 0,
    on_event: Callable | None = None,
):
    last_error = None
    for attempt in range(max_retries):
        try:
            return call(), None
        except Exception as exc:
            if not is_retryable(exc):
                dump = dump_request(debug_dir, provider, exc, request, call_count)
                last_error = _error_dict(exc, attempt, max_retries, dump)
                if on_event:
                    on_event(f"{provider} HTTP {extract_status(exc)} (not retrying): {str(exc)[:160]}"
                             + (f"; request dumped to {dump}" if dump else ""))
                return None, last_error
            last_error = _error_dict(exc, attempt, max_retries)
            if on_event:
                more = attempt < max_retries - 1
                on_event(f"{provider} attempt {attempt + 1}/{max_retries} failed: "
                         f"{type(exc).__name__}: {str(exc)[:140]}" + ("; retrying" if more else "; giving up"))
        if attempt < max_retries - 1:
            backoff_sleep(attempt)
    return None, last_error
