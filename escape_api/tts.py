"""Shared text-to-speech adapter."""
import base64
import hashlib
import inspect
import os
from collections import OrderedDict
from escape_api import llm
from escape_api.settings import cfg

MAX_CHARS = 1600
_CACHE = OrderedDict()
_OVERRIDE = None

def set_override(callback):
    global _OVERRIDE
    _OVERRIDE = callback
    _CACHE.clear()

def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")

async def synthesize(text: str):
    text = (text or "").strip()[:MAX_CHARS]
    if not text:
        raise ValueError("Empty speech text")
    key = hashlib.sha256(text.encode()).hexdigest()
    if key in _CACHE:
        return _CACHE[key]
    if _OVERRIDE is not None:
        result = _OVERRIDE(text)
        if inspect.isawaitable(result):
            result = await result
    else:
        client = llm._get_client() if cfg().llm_enabled else None
        if client is None:
            raise RuntimeError("TTS unavailable")
        voice = os.getenv("LUCILLE_TTS_VOICE", "nova")
        response = await client.audio.speech.create(
            model=os.getenv("LUCILLE_TTS_MODEL", "gpt-4o-mini-tts"),
            voice=voice, input=text, response_format="mp3")
        audio = await response.read() if hasattr(response, "read") else response.content
        result = (audio, "openai", voice)
    if not result[0]:
        raise RuntimeError("TTS returned empty audio")
    _CACHE[key] = result
    if len(_CACHE) > 128:
        _CACHE.popitem(last=False)
    return result
