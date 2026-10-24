"""Regression coverage for OpenAI's asynchronous speech response contract.

The real async speech response has a synchronous read() returning bytes and an
async aread(). Awaiting read() used to raise TypeError, silently removing voice
audio from successful /v1/voice/converse responses.
"""

import pytest

from escape_api import llm, tts
from escape_api.settings import reload_settings


@pytest.mark.asyncio
async def test_async_speech_uses_aread_not_sync_read(monkeypatch):
    monkeypatch.setenv("LUCILLE_V1_LLM_ENABLED", "true")
    reload_settings()
    tts.set_override(None)
    count = []

    class SpeechResponse:
        def read(self):
            raise AssertionError("synchronous read() must not be used")

        async def aread(self):
            return b"ID3-working-mp3"

    class Speech:
        async def create(self, **kwargs):
            count.append(kwargs)
            return SpeechResponse()

    class Audio:
        speech = Speech()

    class FakeClient:
        audio = Audio()

    llm.set_client(FakeClient())
    try:
        first = await tts.synthesize("Hello from Lucille")
        second = await tts.synthesize("Hello from Lucille")
        assert first == (b"ID3-working-mp3", "openai", "nova")
        assert second == first
        assert len(count) == 1
        assert count[0]["response_format"] == "mp3"
        assert count[0]["input"] == "Hello from Lucille"
    finally:
        tts.set_override(None)
        llm.set_client(None)
        reload_settings()


@pytest.mark.asyncio
async def test_speech_response_content_fallback(monkeypatch):
    monkeypatch.setenv("LUCILLE_V1_LLM_ENABLED", "true")
    reload_settings()
    tts.set_override(None)

    class Speech:
        async def create(self, **kwargs):
            return type("Response", (), {"content": b"ID3-fallback"})()

    class Audio:
        speech = Speech()

    class FakeClient:
        audio = Audio()

    llm.set_client(FakeClient())
    try:
        audio, provider, voice = await tts.synthesize("Content property test")
        assert audio == b"ID3-fallback"
        assert provider == "openai"
        assert voice == "nova"
    finally:
        tts.set_override(None)
        llm.set_client(None)
        reload_settings()


@pytest.mark.asyncio
async def test_empty_async_speech_is_rejected(monkeypatch):
    monkeypatch.setenv("LUCILLE_V1_LLM_ENABLED", "true")
    reload_settings()
    tts.set_override(None)

    class SpeechResponse:
        async def aread(self):
            return b""

    class Speech:
        async def create(self, **kwargs):
            return SpeechResponse()

    class Audio:
        speech = Speech()

    class FakeClient:
        audio = Audio()

    llm.set_client(FakeClient())
    try:
        with pytest.raises(RuntimeError, match="empty audio"):
            await tts.synthesize("Blank audio response")
    finally:
        tts.set_override(None)
        llm.set_client(None)
        reload_settings()
