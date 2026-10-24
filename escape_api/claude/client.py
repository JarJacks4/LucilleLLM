"""
A thin async client for the Anthropic Messages API.

Deliberately small: one POST, one streaming POST, retries with backoff, and a
single exception the routers can turn into a clean HTTP answer. No SDK, no
`openai`, no global state besides the injected test double.

    from escape_api.claude import client as claude

    if not claude.available():
        raise claude.ClaudeUnavailable()                 # -> 503 claude_unavailable
    text, usage = await claude.get_client().text(system, messages)
    async for ev in claude.get_client().stream(system, messages, tools=specs):
        ...

Normalised stream events (dicts, so a fake client is trivial to write):

    {"type": "text",     "text": "..."}
    {"type": "tool_use", "id": "toolu_…", "name": "read_mood_summary", "input": {...}}
    {"type": "usage",    "inputTokens": 1234, "outputTokens": 56}
    {"type": "stop",     "reason": "end_turn" | "tool_use" | "max_tokens" | ...}
    {"type": "error",    "code": "overloaded_error", "message": "..."}
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from escape_api.settings import cfg

logger = logging.getLogger(__name__)

# ── models (confirmed from platform.claude.com, Oct 2026) ─────────────────────
# All four: 1M context window, 128K max output.
MODEL_DEFAULT = "claude-opus-5-5"      # the voice of Lucille
MODEL_CHEAP = "claude-sonnet-5-5"      # short / cheap turns
MODEL_FAST = "claude-haiku-5-5"        # titles, summaries, memory extraction, fold-down
MODEL_LONG = "claude-fable-5-1"        # long plan reasoning, opt-in

MODELS = {"default": MODEL_DEFAULT, "cheap": MODEL_CHEAP, "fast": MODEL_FAST, "long": MODEL_LONG}

CONTEXT_WINDOW_TOKENS = 1_000_000
MAX_OUTPUT_TOKENS = 128_000

RETRY_STATUS = (408, 409, 425, 429, 500, 502, 503, 504, 529)


# ── errors ────────────────────────────────────────────────────────────────────

class ClaudeError(Exception):
    """Any failure talking to Anthropic. `code` is safe to show the app."""

    status_code = 502
    code = "claude_error"

    def __init__(self, message: str = "Lucille could not answer just now.", *, code: Optional[str] = None,
                 status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.message = message
        if code:
            self.code = code
        if status_code:
            self.status_code = status_code

    def as_event(self) -> Dict[str, Any]:
        return {"type": "error", "code": self.code, "message": self.message}

    def as_detail(self) -> Dict[str, Any]:
        return {"code": self.code, "message": self.message}


class ClaudeUnavailable(ClaudeError):
    """No ANTHROPIC_API_KEY on this deployment: degrade, never pretend."""

    status_code = 503
    code = "claude_unavailable"

    def __init__(self, message: str = "Lucille (Claude edition) isn't configured on this server yet.") -> None:
        super().__init__(message)


class ClaudeOverloaded(ClaudeError):
    status_code = 503
    code = "claude_overloaded"


# ── availability ──────────────────────────────────────────────────────────────

def api_key() -> str:
    return (os.getenv("ANTHROPIC_API_KEY") or "").strip()


def available() -> bool:
    """True when a real call can be made, or when a test double is installed."""
    return bool(_override) or bool(api_key())


def estimate_tokens(text: str) -> int:
    """Cheap, deterministic estimate (~4 chars per token). Used for budgeting only."""
    return (len(text or "") // 4) + 1


def estimate_messages_tokens(system: str, messages: List[Dict[str, Any]]) -> int:
    total = estimate_tokens(system)
    for m in messages or []:
        c = m.get("content")
        if isinstance(c, str):
            total += estimate_tokens(c)
        elif isinstance(c, list):
            for block in c:
                if isinstance(block, dict):
                    total += estimate_tokens(block.get("text") or json.dumps(block.get("input") or block.get("content") or ""))
        total += 4
    return total


# ── client ────────────────────────────────────────────────────────────────────

class ClaudeClient:
    """One instance per process; httpx client is created lazily and reused."""

    def __init__(self, key: Optional[str] = None) -> None:
        self._key = key or api_key()
        self._http = None

    # -- plumbing --

    def _headers(self) -> Dict[str, str]:
        return {
            "x-api-key": self._key,
            "anthropic-version": cfg().claude_api_version,
            "content-type": "application/json",
            "accept": "application/json",
        }

    async def _client(self):
        import httpx
        if self._http is None or self._http.is_closed:
            s = cfg()
            self._http = httpx.AsyncClient(
                base_url=s.claude_base_url.rstrip("/"),
                timeout=httpx.Timeout(s.claude_timeout_s, connect=s.claude_connect_timeout_s),
            )
        return self._http

    async def aclose(self) -> None:
        if self._http is not None and not self._http.is_closed:
            await self._http.aclose()
        self._http = None

    def _body(self, system: str, messages: List[Dict[str, Any]], *, model: Optional[str] = None,
              max_tokens: Optional[int] = None, tools: Optional[List[Dict[str, Any]]] = None,
              temperature: Optional[float] = None, stream: bool = False) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": model or cfg().claude_model,
            "max_tokens": min(int(max_tokens or cfg().claude_max_tokens), MAX_OUTPUT_TOKENS),
            "messages": messages,
        }
        if system:
            body["system"] = system
        if tools:
            body["tools"] = tools
        if temperature is not None:
            body["temperature"] = temperature
        if stream:
            body["stream"] = True
        return body

    @staticmethod
    def _retry_delay(attempt: int, retry_after: Optional[str]) -> float:
        if retry_after:
            try:
                return min(20.0, float(retry_after))
            except (TypeError, ValueError):
                pass
        return min(8.0, (2 ** attempt) * 0.5) + random.uniform(0, 0.25)

    @staticmethod
    def _raise_for(status: int, payload: Any) -> None:
        err = (payload or {}).get("error") if isinstance(payload, dict) else None
        code = (err or {}).get("type") or f"http_{status}"
        msg = "Lucille could not answer just now."
        if status == 401 or status == 403:
            raise ClaudeUnavailable("Lucille's Claude credentials were rejected.")
        if status == 429:
            raise ClaudeOverloaded("Lucille is busy right now. Try again in a moment.", code="claude_rate_limited")
        if status in (500, 502, 503, 529):
            raise ClaudeOverloaded("Lucille is busy right now. Try again in a moment.", code=code)
        raise ClaudeError(msg, code=code, status_code=502)

    # -- non-streaming --

    async def messages(self, system: str, messages: List[Dict[str, Any]], *, model: Optional[str] = None,
                       max_tokens: Optional[int] = None, tools: Optional[List[Dict[str, Any]]] = None,
                       temperature: Optional[float] = None) -> Dict[str, Any]:
        """Raw Messages response (dict). Retries on 429/5xx/timeouts with backoff."""
        if not self._key:
            raise ClaudeUnavailable()
        import httpx
        body = self._body(system, messages, model=model, max_tokens=max_tokens, tools=tools, temperature=temperature)
        attempts = max(1, cfg().claude_max_retries + 1)
        last: Optional[Exception] = None
        for attempt in range(attempts):
            try:
                http = await self._client()
                r = await http.post("/v1/messages", headers=self._headers(), json=body)
                if r.status_code in RETRY_STATUS and attempt < attempts - 1:
                    await asyncio.sleep(self._retry_delay(attempt, r.headers.get("retry-after")))
                    continue
                if r.status_code >= 400:
                    try:
                        payload = r.json()
                    except Exception:
                        payload = None
                    self._raise_for(r.status_code, payload)
                return r.json()
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last = e
                if attempt < attempts - 1:
                    await asyncio.sleep(self._retry_delay(attempt, None))
                    continue
                logger.warning("claude messages transport failure: %s: %s", type(e).__name__, e)
                raise ClaudeError("Lucille timed out. Try again.", code="claude_timeout") from e
        raise ClaudeError("Lucille timed out. Try again.", code="claude_timeout") from last

    async def text(self, system: str, messages: List[Dict[str, Any]], **kw) -> Tuple[str, Dict[str, Any]]:
        """Convenience: concatenated text blocks + usage."""
        data = await self.messages(system, messages, **kw)
        out = "".join(b.get("text", "") for b in (data.get("content") or []) if b.get("type") == "text")
        return out.strip(), usage_of(data)

    # -- streaming --

    async def stream(self, system: str, messages: List[Dict[str, Any]], *, model: Optional[str] = None,
                     max_tokens: Optional[int] = None, tools: Optional[List[Dict[str, Any]]] = None,
                     temperature: Optional[float] = None) -> AsyncIterator[Dict[str, Any]]:
        """Yield normalised events from a streaming Messages call (SSE)."""
        if not self._key:
            raise ClaudeUnavailable()
        import httpx
        body = self._body(system, messages, model=model, max_tokens=max_tokens, tools=tools,
                          temperature=temperature, stream=True)
        attempts = max(1, cfg().claude_max_retries + 1)
        for attempt in range(attempts):
            started = False
            try:
                http = await self._client()
                async with http.stream("POST", "/v1/messages", headers=self._headers(), json=body) as r:
                    if r.status_code >= 400:
                        raw = await r.aread()
                        try:
                            payload = json.loads(raw.decode("utf-8", "replace"))
                        except Exception:
                            payload = None
                        if r.status_code in RETRY_STATUS and attempt < attempts - 1:
                            await asyncio.sleep(self._retry_delay(attempt, r.headers.get("retry-after")))
                            continue
                        self._raise_for(r.status_code, payload)
                    async for ev in _parse_sse(r.aiter_lines()):
                        started = True
                        yield ev
                return
            except (httpx.TimeoutException, httpx.TransportError) as e:
                if started or attempt >= attempts - 1:
                    logger.warning("claude stream failure: %s: %s", type(e).__name__, e)
                    yield {"type": "error", "code": "claude_timeout", "message": "Lucille's answer was cut off."}
                    return
                await asyncio.sleep(self._retry_delay(attempt, None))


async def _parse_sse(lines: AsyncIterator[str]) -> AsyncIterator[Dict[str, Any]]:
    """Anthropic SSE -> normalised events. Tool input arrives as JSON deltas."""
    tool: Dict[int, Dict[str, Any]] = {}
    async for line in lines:
        line = (line or "").strip()
        if not line or line.startswith(":") or line.startswith("event:"):
            continue
        if not line.startswith("data:"):
            continue
        raw = line[5:].strip()
        if not raw or raw == "[DONE]":
            continue
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            continue
        t = d.get("type")
        if t == "content_block_start":
            block = d.get("content_block") or {}
            if block.get("type") == "tool_use":
                tool[d.get("index", 0)] = {"id": block.get("id"), "name": block.get("name"), "json": ""}
        elif t == "content_block_delta":
            delta = d.get("delta") or {}
            if delta.get("type") == "text_delta" and delta.get("text"):
                yield {"type": "text", "text": delta["text"]}
            elif delta.get("type") == "input_json_delta":
                slot = tool.setdefault(d.get("index", 0), {"id": None, "name": None, "json": ""})
                slot["json"] += delta.get("partial_json") or ""
        elif t == "content_block_stop":
            slot = tool.pop(d.get("index", 0), None)
            if slot and slot.get("name"):
                try:
                    args = json.loads(slot["json"]) if slot["json"].strip() else {}
                except json.JSONDecodeError:
                    args = {}
                yield {"type": "tool_use", "id": slot.get("id"), "name": slot["name"],
                       "input": args if isinstance(args, dict) else {}}
        elif t == "message_delta":
            u = d.get("usage") or {}
            if u:
                yield {"type": "usage", "inputTokens": u.get("input_tokens"), "outputTokens": u.get("output_tokens")}
            reason = (d.get("delta") or {}).get("stop_reason")
            if reason:
                yield {"type": "stop", "reason": reason}
        elif t == "message_start":
            u = ((d.get("message") or {}).get("usage")) or {}
            if u:
                yield {"type": "usage", "inputTokens": u.get("input_tokens"), "outputTokens": u.get("output_tokens")}
        elif t == "error":
            err = d.get("error") or {}
            yield {"type": "error", "code": err.get("type") or "claude_error",
                   "message": "Lucille's answer was interrupted."}


def usage_of(data: Dict[str, Any]) -> Dict[str, Any]:
    u = (data or {}).get("usage") or {}
    return {"inputTokens": u.get("input_tokens"), "outputTokens": u.get("output_tokens"),
            "model": (data or {}).get("model")}


# ── singleton + test double ───────────────────────────────────────────────────

_client: Optional[ClaudeClient] = None
_override: Optional[Any] = None


def get_client():
    """The process client, or whatever a test injected with set_client()."""
    global _client
    if _override is not None:
        return _override
    key = api_key()
    if not key:
        raise ClaudeUnavailable()
    if _client is None or _client._key != key:
        _client = ClaudeClient(key)
    return _client


def set_client(fake: Optional[Any]) -> None:
    """Tests inject an object exposing .messages()/.text()/.stream()."""
    global _override, _client
    _override = fake
    if fake is None:
        _client = None


async def one_shot(system: str, prompt: str, *, model: Optional[str] = None, max_tokens: int = 400,
                   fallback: str = "") -> str:
    """Small helper for titles / summaries / fold-down. Never raises: returns `fallback`."""
    if not available():
        return fallback
    try:
        text, _ = await get_client().text(system, [{"role": "user", "content": prompt}],
                                          model=model or cfg().claude_fast_model, max_tokens=max_tokens)
        return text or fallback
    except Exception as e:  # noqa: BLE001 — a title is never worth failing a turn
        logger.warning("claude one_shot failed (%s): %s", type(e).__name__, e)
        return fallback
