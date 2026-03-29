"""
LucilleLLM - Voice Service (TTS / STT)

Provides speech-to-text (STT) and text-to-speech (TTS) capabilities using
free libraries for MVP: SpeechRecognition (Google free API) for STT and
edge-tts (Microsoft Edge Read Aloud) for TTS.

Designed for provider swappability: TTS_PROVIDER / STT_PROVIDER config
fields allow future upgrades to OpenAI Whisper or Google Cloud Speech.

Graceful degradation: returns meaningful errors if libraries are not installed.

Follows the singleton pattern from other services.
"""

import asyncio
import base64
import io
import logging
from typing import Optional, Tuple

from config import get_config

logger = logging.getLogger(__name__)


class VoiceService:
    """
    Service for speech-to-text and text-to-speech operations.

    STT flow:  audio bytes -> transcribe() -> text string
    TTS flow:  text string -> synthesize() -> MP3 bytes
    """

    def __init__(self):
        config = get_config()
        self._tts_provider = config.TTS_PROVIDER
        self._stt_provider = config.STT_PROVIDER
        self._tts_voice = config.TTS_VOICE
        self._tts_rate = config.TTS_RATE
        self._max_audio_size_mb = config.MAX_AUDIO_SIZE_MB

        # Lazy-check library availability
        self._sr_available = False
        self._edge_tts_available = False

        try:
            import speech_recognition  # noqa: F401
            self._sr_available = True
        except ImportError:
            logger.warning(
                "VoiceService: SpeechRecognition not installed. "
                "STT will be unavailable. Install with: pip install SpeechRecognition"
            )

        try:
            import edge_tts  # noqa: F401
            self._edge_tts_available = True
        except ImportError:
            logger.warning(
                "VoiceService: edge-tts not installed. "
                "TTS will be unavailable. Install with: pip install edge-tts"
            )

        logger.info(
            f"VoiceService initialized - "
            f"stt_provider={self._stt_provider} (available={self._sr_available}), "
            f"tts_provider={self._tts_provider} (available={self._edge_tts_available}), "
            f"voice={self._tts_voice}"
        )

    # ── Properties ────────────────────────────────────────────

    @property
    def stt_available(self) -> bool:
        """Check if STT is available."""
        return self._sr_available

    @property
    def tts_available(self) -> bool:
        """Check if TTS is available."""
        return self._edge_tts_available

    # ── STT: Speech-to-Text ──────────────────────────────────

    def transcribe(
        self, audio_bytes: bytes, audio_format: str = "wav"
    ) -> Tuple[str, Optional[str]]:
        """
        Transcribe audio bytes to text.

        Args:
            audio_bytes: Raw audio file bytes (WAV, FLAC, MP3, etc.)
            audio_format: Audio format hint ("wav", "flac", "mp3", "ogg", "webm")

        Returns:
            Tuple of (transcribed_text, error_message).
            On success: ("transcribed text here", None)
            On failure: ("", "error description")
        """
        # Validate size and content before checking library
        if len(audio_bytes) == 0:
            return "", "Empty audio data"

        size_mb = len(audio_bytes) / (1024 * 1024)
        if size_mb > self._max_audio_size_mb:
            return "", (
                f"Audio too large: {size_mb:.1f}MB exceeds "
                f"{self._max_audio_size_mb}MB limit"
            )

        if not self._sr_available:
            return "", "STT unavailable: SpeechRecognition library not installed"

        import speech_recognition as sr

        recognizer = sr.Recognizer()

        try:
            # Convert audio bytes to AudioData via AudioFile
            audio_io = io.BytesIO(audio_bytes)

            # SpeechRecognition natively supports WAV, AIFF, FLAC
            # For other formats, it can handle them if pydub/ffmpeg is available
            if audio_format.lower() in ("wav", "wave", "flac", "aiff"):
                with sr.AudioFile(audio_io) as source:
                    audio_data = recognizer.record(source)
            else:
                # For MP3, OGG, WEBM — attempt via AudioFile
                # SpeechRecognition uses pydub internally for non-native formats
                try:
                    with sr.AudioFile(audio_io) as source:
                        audio_data = recognizer.record(source)
                except Exception:
                    return "", (
                        f"Unsupported audio format '{audio_format}'. "
                        "Please send WAV or FLAC. For MP3/OGG support, "
                        "install pydub and ffmpeg."
                    )

            # Use Google's free Web Speech API (no key required)
            text = recognizer.recognize_google(audio_data)
            logger.info(f"STT transcription successful: {len(text)} chars")
            return text, None

        except sr.UnknownValueError:
            return "", (
                "Could not understand the audio. "
                "Please speak clearly and try again."
            )
        except sr.RequestError as e:
            logger.error(f"STT API request failed: {e}")
            return "", f"Speech recognition service error: {e}"
        except Exception as e:
            logger.error(f"STT transcription failed: {e}")
            return "", f"Transcription failed: {str(e)}"

    # ── TTS: Text-to-Speech ──────────────────────────────────

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        rate: Optional[str] = None,
    ) -> Tuple[bytes, Optional[str]]:
        """
        Synthesize text to speech (MP3 bytes).

        Args:
            text: Text to convert to speech
            voice: Voice name (e.g. "en-US-AriaNeural"). None = use config default.
            rate: Speech rate (e.g. "+10%", "-5%"). None = use config default.

        Returns:
            Tuple of (mp3_bytes, error_message).
            On success: (b"mp3 data...", None)
            On failure: (b"", "error description")
        """
        if not self._edge_tts_available:
            return b"", "TTS unavailable: edge-tts library not installed"

        if not text or not text.strip():
            return b"", "No text provided for synthesis"

        # Truncate very long text to avoid abuse
        if len(text) > 5000:
            text = text[:5000]
            logger.warning("TTS text truncated to 5000 characters")

        import edge_tts

        voice = voice or self._tts_voice
        rate = rate or self._tts_rate

        try:
            communicate = edge_tts.Communicate(text, voice, rate=rate)
            mp3_buffer = io.BytesIO()

            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    mp3_buffer.write(chunk["data"])

            mp3_bytes = mp3_buffer.getvalue()

            if not mp3_bytes:
                return b"", "TTS produced empty audio output"

            logger.info(
                f"TTS synthesis successful: {len(text)} chars -> "
                f"{len(mp3_bytes)} bytes MP3"
            )
            return mp3_bytes, None

        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            return b"", f"Speech synthesis failed: {str(e)}"

    # ── Utility ──────────────────────────────────────────────

    def decode_base64_audio(self, b64_string: str) -> Tuple[bytes, Optional[str]]:
        """
        Decode a base64-encoded audio string.

        Args:
            b64_string: Base64-encoded audio data (supports data URL prefix)

        Returns:
            Tuple of (audio_bytes, error_message)
        """
        if not b64_string:
            return b"", "No audio data provided"

        try:
            # Handle data URL prefix (e.g. "data:audio/wav;base64,...")
            if "," in b64_string and b64_string.startswith("data:"):
                b64_string = b64_string.split(",", 1)[1]

            audio_bytes = base64.b64decode(b64_string)

            if not audio_bytes:
                return b"", "Decoded audio is empty"

            return audio_bytes, None
        except Exception as e:
            return b"", f"Invalid base64 audio data: {str(e)}"

    def encode_audio_base64(self, audio_bytes: bytes) -> str:
        """Encode audio bytes to base64 string."""
        return base64.b64encode(audio_bytes).decode("utf-8")

    def estimate_audio_duration_ms(self, mp3_bytes: bytes) -> int:
        """
        Estimate MP3 audio duration in milliseconds.

        Uses a rough estimate based on average edge-tts bitrate (~48kbps).
        For exact duration, mutagen would be needed.
        """
        if not mp3_bytes:
            return 0
        # edge-tts output is ~48kbps on average
        # duration_seconds = size_bytes * 8 / bitrate_bps
        estimated_bits = len(mp3_bytes) * 8
        duration_ms = int((estimated_bits / 48000) * 1000)
        return duration_ms


# ── Singleton ─────────────────────────────────────────────

_voice_service: Optional[VoiceService] = None


def get_voice_service() -> VoiceService:
    """Get or create VoiceService singleton."""
    global _voice_service
    if _voice_service is None:
        _voice_service = VoiceService()
    return _voice_service
