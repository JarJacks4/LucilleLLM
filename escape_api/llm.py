"""
Cost-aware LLM helper for the v1 API.

Rules that keep the bill small:
  * one small model (LUCILLE_V1_MODEL, default gpt-4o-mini), JSON mode, short max_tokens
  * every call has a deterministic fallback, so a timeout or missing key never breaks a screen
  * results are stored on the document they belong to (an entry's reflection is generated
    once and re-read for free), and identical cohort prompts are cached in memory
  * callers check a per-user daily quota first (core.check_quota)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Dict, Optional

from escape_api.settings import cfg

logger = logging.getLogger(__name__)

_client = None
_cache: Dict[str, Dict[str, Any]] = {}
_CACHE_MAX = 512

LUCILLE_SYSTEM = (
    "You are Lucille, the AI self-care companion inside the Escape app. Warm, brief, plain words. "
    "You are not a therapist: never diagnose, never name disorders, never promise outcomes, never give "
    "medical advice. Reflect what the person wrote, validate it, and end somewhere constructive. "
    "Never invent facts about the person. Always answer with a single JSON object only."
)


def _get_client():
    global _client
    if _client is None:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            return None
        from openai import AsyncOpenAI
        _client = AsyncOpenAI(api_key=key, timeout=cfg().llm_timeout_s, max_retries=1)
    return _client


def set_client(client) -> None:
    """Tests inject a fake client."""
    global _client
    _client = client


async def complete_json(
    task: str,
    user_content: str,
    fallback: Dict[str, Any],
    cache_key: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Return parsed JSON from the model, or `fallback` (with _source='fallback')."""
    if cache_key:
        ck = hashlib.sha256(cache_key.encode()).hexdigest()
        if ck in _cache:
            return dict(_cache[ck], _source="cache")
    client = _get_client() if cfg().llm_enabled else None
    if client is None:
        return dict(fallback, _source="fallback")
    try:
        resp = await client.chat.completions.create(
            model=cfg().llm_model,
            temperature=0.6,
            max_tokens=max_tokens or cfg().llm_max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": LUCILLE_SYSTEM + " Task: " + task},
                {"role": "user", "content": user_content[:6000]},
            ],
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        if not isinstance(data, dict):
            raise ValueError("non-object JSON")
        out = {**fallback, **{k: v for k, v in data.items() if v not in (None, "")}}
        usage = getattr(resp, "usage", None)
        out["_source"] = "llm"
        out["_tokens"] = getattr(usage, "total_tokens", None)
        if cache_key:
            if len(_cache) > _CACHE_MAX:
                _cache.clear()
            _cache[ck] = {k: v for k, v in out.items() if not k.startswith("_")}
        return out
    except Exception as e:
        logger.warning(f"v1 LLM call failed, using fallback: {type(e).__name__}: {e}")
        return dict(fallback, _source="fallback")
