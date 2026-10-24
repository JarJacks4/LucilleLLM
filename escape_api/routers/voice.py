"""/v1/voice — speech to text for the voice journal. The speech-to-text key lives only on the server."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from escape_api import core, llm, tts
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

CONVERSE_DAILY_LIMIT = 20
DISTRESS_REPLY = ("I'm glad you told me. If you might act on these thoughts, "
                  "call or text 988 in the US or contact local emergency services.")

class SpeakIn(BaseModel):
    text: str = Field(min_length=1, max_length=2000)

async def _read_audio(file: UploadFile):
    name = (file.filename or "").lower()
    ctype = (file.content_type or "").lower()
    if not name.endswith(_ALLOWED):
        if not ctype.startswith("audio/"):
            raise HTTPException(415, "Unsupported audio format.")
        name = "audio.m4a"
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty recording.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Recording is too long.")
    return name, data

async def _whisper(name: str, data: bytes) -> str:
    client = llm._get_client() if cfg().llm_enabled else None
    if client is None:
        raise HTTPException(503, "Transcription unavailable.")
    try:
        result = await client.audio.transcriptions.create(model="whisper-1", file=(name, data))
        return (getattr(result, "text", "") or "").strip()
    except Exception:
        raise HTTPException(502, "Transcription failed.")

@router.post("/speak")
async def speak(body: SpeakIn, c: Caller = Depends(caller)):
    if not body.text.strip():
        raise HTTPException(422, "Text cannot be blank.")
    if not core.check_quota(c.repo, c.uid, c.tz, "speak", 40):
        raise HTTPException(429, "Daily voice limit reached.")
    try:
        audio, provider, voice = await tts.synthesize(body.text)
    except Exception as exc:
        # Do not log user speech text or the upstream exception body.
        logger.warning("speak TTS unavailable: %s", type(exc).__name__)
        raise HTTPException(503, "Voice generation unavailable.")
    return {"audio": tts.b64(audio), "audioMime": "audio/mpeg",
            "voiceProvider": provider, "voice": voice, "clonedVoice": False}

@router.post("/converse")
async def converse(file: UploadFile = File(...), session_id: str = Form(""),
                   c: Caller = Depends(caller)):
    require_consent(c, "voice")
    require_consent(c, "ai_reflection")
    name, data = await _read_audio(file)
    if not core.check_quota(c.repo, c.uid, c.tz, "converse", CONVERSE_DAILY_LIMIT):
        raise HTTPException(429, "Daily conversation limit reached.")
    transcript = await _whisper(name, data)
    if not transcript:
        raise HTTPException(400, "No speech recognized.")
    distress = bool(core.safety_check(transcript).get("crisis"))
    if distress:
        reply = DISTRESS_REPLY
    else:
        fallback = {"reply": "Thank you for sharing. Would you like to tell me more?"}
        result = await llm.complete_json(
            task="Write a brief supportive non-clinical reply as JSON with key reply.",
            user_content="<entry>" + transcript[:4000] + "</entry>",
            fallback=fallback, max_tokens=160)
        reply = str(result.get("reply") or fallback["reply"])
    try:
        audio, provider, voice = await tts.synthesize(reply)
        encoded = tts.b64(audio)
    except Exception as exc:
        # Preserve captions-only fallback, but make TTS failures visible in
        # server logs without recording sensitive transcript/reply content.
        logger.warning("converse TTS unavailable: %s", type(exc).__name__)
        encoded = None
    return {"transcript": transcript, "reply": reply, "audio": encoded,
            "audioMime": "audio/mpeg" if encoded else None,
            "distress": distress, "sessionId": session_id or core.new_id("voice")}
