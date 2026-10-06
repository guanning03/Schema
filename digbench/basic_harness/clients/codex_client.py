from __future__ import annotations

import json
import time

import httpx

from clients import codex_auth as cxa
from clients.openai_client import OPENAI_MAX_CONTEXT, OpenAIPolicy


class _Attr(dict):
    def __getattr__(self, k):
        if k not in self:
            return None
        v = self[k]
        if isinstance(v, dict): return _Attr(v)
        if isinstance(v, list): return [_Attr(x) if isinstance(x, dict) else x for x in v]
        return v


class _Resp:
    def __init__(self, output, usage, model, status, output_text):
        self.output = output; self.usage = usage; self.model = model
        self.status = status; self.output_text = output_text


class _Responses:
    def __init__(self, auth, timeout): self._auth = auth; self._timeout = timeout
    def create(self, **kw):
        auth = self._auth
        body = {
            "model": kw["model"], "input": list(kw["input"]),
            "stream": True, "store": False,
            "prompt_cache_key": auth._session_uuid,
            "instructions": kw.get("instructions", ""),
            "include": kw.get("include", ["reasoning.encrypted_content"]),
        }
        if kw.get("tools"): body["tools"] = kw["tools"]; body["tool_choice"] = kw.get("tool_choice", "auto")
        if "parallel_tool_calls" in kw: body["parallel_tool_calls"] = kw["parallel_tool_calls"]
        if kw.get("reasoning"): body["reasoning"] = kw["reasoning"]
        return _stream(auth, body)


def _stream(auth, body):
    refreshed = False
    while True:
        items = []
        completed_output = None
        usage = {}; model = None; status = None; text_parts = []
        try:
            with httpx.Client(timeout=auth.timeout) as client:
                with client.stream("POST", cxa.BASE_URL, json=body, headers=auth.headers()) as resp:
                    if resp.status_code >= 400:
                        bt = resp.read().decode("utf-8", "replace")
                        lim = cxa.classify_limit(resp.status_code, resp.headers, bt)
                        if lim is not None: raise cxa.CodexLimitError(lim[0], lim[1], bt[:300])
                        raise httpx.HTTPStatusError(f"{resp.status_code}: {bt[:800]}", request=resp.request, response=resp)
                    for line in resp.iter_lines():
                        if not line or not line.startswith("data:"): continue
                        data = line[len("data:"):].strip()
                        if not data or data == "[DONE]": continue
                        try: ev = json.loads(data)
                        except json.JSONDecodeError: continue
                        et = ev.get("type", "")
                        if et == "response.output_text.delta": text_parts.append(ev.get("delta","") or "")
                        elif et == "response.output_item.done":
                            it = ev.get("item")
                            if isinstance(it, dict): items.append(it)
                        elif et in ("response.completed", "response.incomplete"):
                            r = ev.get("response") or {}
                            usage = r.get("usage") or {}; model = r.get("model"); status = r.get("status")
                            if r.get("output"): completed_output = r["output"]
                            if r.get("output_text"): text_parts = [r["output_text"]]
                        elif et == "error":
                            msg = str(ev.get("message") or ev)
                            if cxa.LIMIT_MSG_RE.search(msg):
                                secs = cxa.find_reset_seconds(ev)
                                raise cxa.CodexLimitError("limit", time.time()+secs if secs is not None else None, msg[:300])
                            raise cxa.CodexAuthError(f"codex stream error: {msg}")
            src = completed_output if completed_output else items
            out = [_Attr(x) if isinstance(x, dict) else x for x in (src or [])]
            return _Resp(out, _Attr(usage), model, status, "".join(text_parts).strip())
        except cxa.CodexLimitError as e:
            if auth.pool is None: raise cxa.CodexAuthError(f"codex usage limit: {e}") from e
            auth.rotate_account(e.kind, e.reset_at); refreshed = False; continue
        except httpx.HTTPStatusError as e:
            if e.response is not None and e.response.status_code == 401 and not refreshed:
                refreshed = True
                if auth.refresh(): continue
                if auth.pool is not None and len(auth.pool) > 1:
                    auth.rotate_account("auth-failed", None); refreshed = False; continue
            detail = ""
            try: detail = e.response.text[:400] if e.response is not None else ""
            except Exception: pass
            raise cxa.CodexAuthError(f"codex HTTP {e}; {detail}") from e


class CodexPolicy(OpenAIPolicy):
    def __init__(self, *, model, effort, max_tokens, timeout, max_retries, include_thoughts=True):
        auth = cxa.CodexAuth(timeout=timeout)
        if not auth.available():
            raise SystemExit("no codex login (~/.codex-arc-agent*/auth.json or ~/.codex/auth.json)")
        self._auth = auth
        client = type("C", (), {"responses": _Responses(auth, timeout)})()
        super().__init__(model=model, effort=effort, max_tokens=max_tokens,
                         timeout=timeout, max_retries=max_retries, pricing={},
                         include_thoughts=include_thoughts,
                         base_url=cxa.BASE_URL, client=client)
        self.model_max_context = OPENAI_MAX_CONTEXT.get("openai.gpt-5.5", 272_000)
