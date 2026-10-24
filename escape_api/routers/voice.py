"""/v1/voice — speech to text for the voice journal. The speech-to-text key lives only on the server."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from escape_api import core, llm
from escape_api.deps import Caller, caller, require_consent
from escape_api.settings import cfg

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/voice", tags=["v1 · voice"])

MAX_BYTES = 12 * 1024 * 1024        # ~10 minutes of AAC
DAILY_LIMIT = 40
_ALLOWED = (".m4a", ".mp3", ".mp4", ".wav", ".webm", ".ogg", ".aac", ".caf")


@router.post("/transcribe", summary="Turn a voice journal recording into text (audio is not stored)")
async def transcribe(file: UploadFile = File(...), c: Caller = Depends(caller)):
    require_consent(c, "voice")
    name = (file.filename or "").lower()
    ctype = (file.content_type or "").lower()
    if not name.endswith(_ALLOWED):
        # Some clients send no filename or a generic one; trust an audio content type, otherwise refuse.
        if ctype.startswith(("audio/", "video/mp4")) or (not name and ctype in ("", "application/octet-stream")):
            name = "audio.m4a"
        else:
            raise HTTPException(415, "Unsupported audio format.")
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty recording.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Recording is too long. Keep it under about 10 minutes.")
    if not core.check_quota(c.repo, c.uid, c.tz, "transcribe", DAILY_LIMIT):
        raise HTTPException(429, "Daily voice limit reached. Try again tomorrow.")
    client = llm._get_client() if cfg().llm_enabled else None
    if client is None:
        raise HTTPException(503, "Voice transcription is not available right now.")
    try:
        r = await client.audio.transcriptions.create(model="whisper-1", file=(name, data))
    except Exception as e:  # never echo provider errors back to the app
        logger.warning("transcribe failed: %s: %s", type(e).__name__, e)
        raise HTTPException(502, "Could not transcribe that recording.")
    return {"text": (getattr(r, "text", "") or "").strip()}
