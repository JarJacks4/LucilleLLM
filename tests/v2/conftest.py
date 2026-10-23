"""
Fixtures for the Claude edition (/v2) tests.

Same shape as tests/v1: a FastAPI TestClient over a MemoryRepo, a fake caller, and
no network — the Anthropic client is a scripted double, and the memory / firebase /
user services are injected fakes so nothing touches Firestore or OpenAI.
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from escape_api import register  # noqa: E402
from escape_api.claude import client as claude  # noqa: E402
from escape_api.claude import context as ctx_mod  # noqa: E402
from escape_api.repo import MemoryRepo, set_repo  # noqa: E402
from escape_api.settings import reload_settings  # noqa: E402

H = {"X-Test-Uid": "u1", "X-Timezone": "America/New_York"}


async def fake_user(request: Request):
    uid = request.headers.get("X-Test-Uid")
    if not uid:
        raise HTTPException(401, "no test uid")
    return {"uid": uid, "role": "admin"} if uid == "admin" else {"uid": uid}


# ── doubles ──────────────────────────────────────────────────────────────────

class FakeClaude:
    """
    Scripted Anthropic client. `scripts` is a list of event lists, one per
    streaming round; `texts` feeds the non-streaming helper (titles, fold-down,
    plan authoring).
    """

    def __init__(self, scripts: Optional[List[List[Dict[str, Any]]]] = None,
                 texts: Optional[List[str]] = None) -> None:
        self.scripts = [list(s) for s in (scripts or [])]
        self.texts = list(texts or [])
        self.stream_calls: List[Dict[str, Any]] = []
        self.text_calls: List[Dict[str, Any]] = []

    async def stream(self, system, messages, tools=None, model=None, max_tokens=None, temperature=None):
        self.stream_calls.append({"system": system, "messages": messages, "tools": tools, "model": model})
        script = self.scripts.pop(0) if self.scripts else [
            {"type": "text", "text": "I'm here."},
            {"type": "usage", "inputTokens": 10, "outputTokens": 4},
            {"type": "stop", "reason": "end_turn"},
        ]
        for ev in script:
            yield ev

    async def text(self, system, messages, model=None, max_tokens=None, temperature=None):
        self.text_calls.append({"system": system, "messages": messages, "model": model})
        out = self.texts.pop(0) if self.texts else "A gentle check in"
        return out, {"inputTokens": 5, "outputTokens": 3, "model": model}

    async def messages(self, system, messages, **kw):
        text, usage = await self.text(system, messages, **kw)
        return {"content": [{"type": "text", "text": text}], "usage": {"input_tokens": 1, "output_tokens": 1}}


class FakeMemory:
    """Stands in for memory_service.MemoryService (same methods, in-memory)."""

    def __init__(self, rows: Optional[List[Dict[str, Any]]] = None) -> None:
        self.rows = list(rows or [])
        self.stored: List[Dict[str, Any]] = []

    def get_memories(self, user_id, memory_type=None, limit=50):
        return list(self.rows)[:limit]

    def store_memory(self, user_id, memory_data):
        memory_data = dict(memory_data, memory_id=f"m{len(self.stored) + 1}", user_id=user_id)
        self.stored.append(memory_data)
        self.rows.append(memory_data)
        return memory_data["memory_id"]

    def format_memories_for_prompt(self, memories):
        if not memories:
            return ""
        return "--- USER MEMORIES ---\n" + "\n".join(f"[Known] {m['content']}" for m in memories) \
               + "\n--- END MEMORIES ---"


class FakeUsers:
    def __init__(self, profile: Optional[Dict[str, Any]] = None) -> None:
        self.profile = profile

    def get_user_profile(self, user_id):
        return self.profile

    def format_profile_for_prompt(self, profile):
        return "Prefers mornings. Lives in Lisbon." if profile else ""


class FakeFirebase:
    """A tiny chat_sessions store, so 'memory since inception' can be asserted."""

    def __init__(self) -> None:
        self.db = object()
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def get_chat_session(self, session_id):
        return self.sessions.get(session_id)

    def store_chat_session(self, session_id, messages, summary=""):
        self.sessions[session_id] = {
            "session_id": session_id, "summary": summary,
            "messages": [{"type": "human" if m.__class__.__name__ == "HumanMessage" else "ai",
                          "content": m.content} for m in messages]}
        return True

    def update_chat_session(self, session_id, messages, summary=""):
        return self.store_chat_session(session_id, messages, summary)

    def delete_chat_session(self, session_id):
        return self.sessions.pop(session_id, None) is not None

    def link_session_to_user(self, session_id, user_id):
        self.sessions.setdefault(session_id, {}).update({"user_id": user_id})
        return True

    def get_user_sessions(self, user_id, limit=50):
        return [dict(s, updated_at="2026-01-01T00:00:00Z") for s in self.sessions.values()
                if s.get("user_id") == user_id]


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def fakes():
    return {"memory": FakeMemory(), "users": FakeUsers(), "firebase": FakeFirebase()}


@pytest.fixture
def client(monkeypatch, fakes):
    monkeypatch.setenv("LUCILLE_EDITION", "claude")
    monkeypatch.setenv("ENFORCE_CONSENT", "true")
    monkeypatch.setenv("WEATHER_PROVIDER", "none")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("LUCILLE_PLAN_DOCS_BUCKET", raising=False)
    reload_settings()
    repo = MemoryRepo()
    set_repo(repo)
    ctx_mod.set_services(memory=fakes["memory"], users=fakes["users"], firebase=fakes["firebase"])
    app = FastAPI()
    register(app)
    from escape_api import deps
    app.dependency_overrides[deps.get_current_user] = fake_user
    c = TestClient(app)
    c.repo = repo
    c.fakes = fakes
    yield c
    set_repo(None)
    ctx_mod.set_services()
    claude.set_client(None)


@pytest.fixture
def lucille(client):
    """The client with a scripted Claude installed. Returns (client, fake)."""
    fake = FakeClaude()
    claude.set_client(fake)
    return client, fake


def consent(client, **flags):
    body = flags or {"wellbeing_data": True, "ai_reflection": True, "personalization": True, "voice": True}
    r = client.put("/v1/privacy/consents", json={"flags": body}, headers=H)
    assert r.status_code == 200, r.text


def events(response) -> List[Dict[str, Any]]:
    """Parse an SSE body into event dicts."""
    return [json.loads(line[5:]) for line in response.text.splitlines() if line.startswith("data:")]


def types(evs) -> List[str]:
    return [e["type"] for e in evs]


def first(evs, kind: str) -> Optional[Dict[str, Any]]:
    return next((e for e in evs if e["type"] == kind), None)


def text_of(evs) -> str:
    return "".join(e.get("text", "") for e in evs if e["type"] == "delta")
