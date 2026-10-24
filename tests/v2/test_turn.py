"""POST /v2/lucille/turn — SSE event sequence, tools, approvals, consent, quota, safety."""

import json

from escape_api.claude import client as claude
from escape_api.repo import user_path

from tests.v2.conftest import H, consent, events, first, text_of, types


def _say(text, tools=None, stop="end_turn"):
    """One streaming round."""
    evs = [{"type": "text", "text": text}] if text else []
    for t in (tools or []):
        evs.append({"type": "tool_use", "id": f"toolu_{t[0]}", "name": t[0], "input": t[1]})
    evs.append({"type": "usage", "inputTokens": 100, "outputTokens": 20})
    evs.append({"type": "stop", "reason": "tool_use" if tools else stop})
    return evs


# ── gates ────────────────────────────────────────────────────────────────────

def test_turn_requires_auth(client):
    assert client.post("/v2/lucille/turn", json={"message": "hi"}).status_code == 401


def test_turn_requires_ai_reflection_consent(lucille):
    c, _ = lucille
    r = c.post("/v2/lucille/turn", json={"message": "hi"}, headers=H)
    assert r.status_code == 403 and r.json()["detail"]["purpose"] == "ai_reflection"


def test_turn_503_without_an_anthropic_key(client):
    consent(client)
    claude.set_client(None)
    r = client.post("/v2/lucille/turn", json={"message": "hi"}, headers=H)
    assert r.status_code == 503
    assert r.json()["detail"]["code"] == "claude_unavailable"


def test_turn_quota(lucille, monkeypatch):
    from escape_api.settings import reload_settings
    c, fake = lucille
    consent(c)
    monkeypatch.setenv("QUOTA_CLAUDE_TURN_PER_DAY", "1")
    reload_settings()
    assert c.post("/v2/lucille/turn", json={"message": "one"}, headers=H).status_code == 200
    r = c.post("/v2/lucille/turn", json={"message": "two"}, headers=H)
    assert r.status_code == 429 and r.json()["detail"]["code"] == "quota_exceeded"


def test_empty_message_rejected(lucille):
    c, _ = lucille
    consent(c)
    assert c.post("/v2/lucille/turn", json={"message": "   "}, headers=H).status_code == 422


# ── the happy path ───────────────────────────────────────────────────────────

def test_event_sequence_session_status_delta_disclosure_done(lucille):
    c, fake = lucille
    consent(c)
    fake.scripts = [_say("That sounds like a long night. ") + []]
    fake.scripts = [[{"type": "text", "text": "That sounds like a long night. "},
                     {"type": "text", "text": "What helped last time?"},
                     {"type": "usage", "inputTokens": 120, "outputTokens": 18},
                     {"type": "stop", "reason": "end_turn"}]]
    fake.texts = ["A long night"]
    r = c.post("/v2/lucille/turn", json={"message": "I can't sleep", "sessionId": None}, headers=H)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    evs = events(r)
    seq = types(evs)
    assert seq[0] == "session" and seq[1] == "status" and seq[-1] == "done"
    assert set(seq) <= {"session", "status", "delta", "artifact", "action", "plan", "approval",
                        "safety", "disclosure", "done", "error"}
    assert first(evs, "session")["title"] == "A long night"
    assert first(evs, "status")["kind"] == "thinking"
    assert text_of(evs) == "That sounds like a long night. What helped last time?"
    assert first(evs, "disclosure")["text"].startswith("Lucille is an AI")   # first ever turn
    done = first(evs, "done")
    assert done["messageId"] and done["usage"]["outputTokens"] == 18
    # the turn is stored for both editions to read
    thread = c.get("/v2/lucille/threads", headers=H).json()["items"][0]
    assert thread["title"] == "A long night" and thread["messages"] == 2
    assert len(c.fakes["firebase"].sessions) == 1


def test_second_turn_reuses_the_thread_and_its_history(lucille):
    c, fake = lucille
    consent(c)
    fake.texts = ["Sleep"]
    sid = first(events(c.post("/v2/lucille/turn", json={"message": "I can't sleep"}, headers=H)),
                "session")["sessionId"]
    fake.scripts = [_say("Still awake?")]
    evs = events(c.post("/v2/lucille/turn", json={"message": "still awake", "sessionId": sid}, headers=H))
    assert first(evs, "session")["sessionId"] == sid
    assert first(evs, "session")["title"] == "Sleep"             # not re-titled
    # the second call saw the first turn verbatim
    sent = fake.stream_calls[-1]["messages"]
    assert sent[0]["content"] == "I can't sleep" and sent[-1]["content"] == "still awake"
    assert "disclosure" not in types(evs)                        # cadence, not every turn


def test_status_lines_only_describe_what_happened(lucille):
    c, fake = lucille
    consent(c)
    c.post("/v1/mood/checkins", json={"word": "Restless"}, headers=H)
    fake.scripts = [_say("Let me look.", tools=[("read_mood_summary", {"range": "7d"})]),
                    _say("Your last check-in was restless, which is why I'd keep tonight light.")]
    evs = events(c.post("/v2/lucille/turn", json={"message": "how have I been?"}, headers=H))
    labels = [e["label"] for e in evs if e["type"] == "status"]
    kinds = {e["kind"] for e in evs if e["type"] == "status"}
    assert "Thinking…" in labels
    assert any(l.startswith("Read your last 1 check-in") for l in labels)
    assert kinds <= {"thinking", "reading", "writing", "searching", "drafting", "checking"}
    # the tool result really reached the model
    results = [m for m in fake.stream_calls[-1]["messages"] if isinstance(m["content"], list)]
    payload = json.loads(results[-1]["content"][0]["content"])
    assert payload["checkins"] == 1 and payload["latest"]["word"] == "Restless"


def test_tools_offered_depend_on_consent(lucille):
    c, fake = lucille
    consent(c, wellbeing_data=True, ai_reflection=True)          # no personalization
    c.post("/v2/lucille/turn", json={"message": "hi"}, headers=H)
    names = {t["name"] for t in fake.stream_calls[0]["tools"]}
    assert "draft_plan" not in names and "adjust_plan" not in names
    assert {"read_mood_summary", "make_chart", "suggest_route"} <= names


def test_chart_tool_emits_an_artifact_with_real_or_empty_data(lucille):
    c, fake = lucille
    consent(c)
    fake.scripts = [_say("", tools=[("make_chart", {"metric": "mood", "range": "7d"})]),
                    _say("There isn't enough yet to draw a shape I'd trust.")]
    evs = events(c.post("/v2/lucille/turn", json={"message": "show me my mood"}, headers=H))
    art = first(evs, "artifact")
    assert art["kind"] == "chart" and art["mime"] == "image/svg+xml"
    assert art["empty"] is True and "Not enough history yet" in art["svg"]
    assert any("not enough yet" in e["label"] for e in evs if e["type"] == "status")


def test_route_suggestion_is_validated(lucille):
    c, fake = lucille
    consent(c)
    fake.scripts = [_say("", tools=[("suggest_route", {"route": "NotARealPage", "label": "Open"}),
                                    ("suggest_route", {"route": "BoxBreathingGoalPage",
                                                       "label": "Box breathing"})]),
                    _say("Four in, four out — because you said your chest feels tight.")]
    evs = events(c.post("/v2/lucille/turn", json={"message": "help me breathe"}, headers=H))
    actions = [e for e in evs if e["type"] == "action"]
    assert len(actions) == 1
    assert actions[0]["route"] == "BoxBreathingGoalPage" and actions[0]["url"].startswith("escape://")


# ── approvals ────────────────────────────────────────────────────────────────

def test_write_tool_waits_for_approval_and_writes_nothing_until_allowed(lucille):
    c, fake = lucille
    consent(c)
    fake.scripts = [_say("", tools=[("draft_plan", {"goals": ["sleep_better"], "days": 3,
                                                    "minutesPerDay": 15})]),
                    _say("I'd like to write you a three-day week. Say the word.")]
    evs = events(c.post("/v2/lucille/turn", json={"message": "make me a plan"}, headers=H))
    ap = first(evs, "approval")
    assert ap["label"].startswith("Lucille wants to write you a new plan")
    assert "replaces your current plan" in ap["detail"] and ap["destructive"] is False
    assert first(evs, "plan") is None                              # nothing written yet
    assert c.get("/v2/plans/active", headers=H).json()["plan"] is None
    assert first(evs, "done")["awaitingApproval"] is True

    allowed = c.post("/v2/lucille/approve", json={"id": ap["id"], "allow": True}, headers=H)
    assert allowed.status_code == 200
    j = allowed.json()
    assert j["written"] is True
    plan_ev = next(e for e in j["events"] if e["type"] == "plan")
    assert plan_ev["days"] == 3
    plan = c.get("/v2/plans/active", headers=H).json()["plan"]
    assert plan["id"] == plan_ev["planId"] and len(plan["items"]) >= 3
    # one-shot: the same approval can't be replayed
    assert c.post("/v2/lucille/approve", json={"id": ap["id"], "allow": True},
                  headers=H).status_code == 404


def test_declining_an_approval_writes_nothing(lucille):
    c, fake = lucille
    consent(c)
    fake.scripts = [_say("", tools=[("draft_plan", {"goals": ["focus"]})]), _say("Only if you want.")]
    ap = first(events(c.post("/v2/lucille/turn", json={"message": "plan?"}, headers=H)), "approval")
    r = c.post("/v2/lucille/approve", json={"id": ap["id"], "allow": False}, headers=H)
    assert r.json() == {"id": ap["id"], "allowed": False, "events": [], "written": False}
    assert c.get("/v2/plans/active", headers=H).json()["plan"] is None


def test_expired_approval_is_refused(lucille):
    c, fake = lucille
    consent(c)
    fake.scripts = [_say("", tools=[("draft_plan", {"goals": ["focus"]})]), _say("ok")]
    ap = first(events(c.post("/v2/lucille/turn", json={"message": "plan?"}, headers=H)), "approval")
    from escape_api.claude.tools import APPROVALS
    c.repo.set(user_path("u1", APPROVALS, ap["id"]), {"expiresAt": "2000-01-01T00:00:00Z"}, merge=True)
    r = c.post("/v2/lucille/approve", json={"id": ap["id"], "allow": True}, headers=H)
    assert r.status_code == 404 and r.json()["detail"]["code"] == "approval_not_found"
    assert c.get("/v2/plans/active", headers=H).json()["plan"] is None


def test_plan_tool_blocked_without_personalization(lucille):
    c, fake = lucille
    consent(c, wellbeing_data=True, ai_reflection=True)
    # even if the model asks for it, consent decides
    fake.scripts = [_say("", tools=[("draft_plan", {"goals": ["focus"]})]), _say("I can't do that yet.")]
    evs = events(c.post("/v2/lucille/turn", json={"message": "plan"}, headers=H))
    assert first(evs, "approval") is None
    results = [m for m in fake.stream_calls[-1]["messages"] if isinstance(m["content"], list)]
    payload = json.loads(results[-1]["content"][0]["content"])
    assert payload["error"] == "consent_required" and payload["purpose"] == "personalization"


# ── safety ───────────────────────────────────────────────────────────────────

def test_distress_never_reaches_the_model(lucille):
    c, fake = lucille
    consent(c)

    async def explode(*a, **kw):
        raise AssertionError("distress text must not be sent to Claude")
        yield {}

    fake.stream = explode
    evs = events(c.post("/v2/lucille/turn", json={"message": "I want to kill myself"}, headers=H))
    saf = first(evs, "safety")
    assert saf["level"] == "urgent" and saf["resources"]
    assert "988" in json.dumps(saf["resources"])
    assert text_of(evs) and "988" in text_of(evs)
    assert first(evs, "disclosure") and types(evs)[-1] == "done"
    assert first(evs, "done")["safety"] == "urgent"


# ── errors ───────────────────────────────────────────────────────────────────

def test_stream_error_becomes_an_error_event(lucille):
    c, fake = lucille
    consent(c)
    fake.scripts = [[{"type": "error", "code": "overloaded_error", "message": "busy"},
                     {"type": "stop", "reason": "end_turn"}]]
    evs = events(c.post("/v2/lucille/turn", json={"message": "hello"}, headers=H))
    err = first(evs, "error")
    assert err["code"] == "overloaded_error"
    assert types(evs)[-1] == "done"


# ── threads ──────────────────────────────────────────────────────────────────

def test_thread_list_read_delete_and_isolation(lucille):
    c, fake = lucille
    consent(c)
    fake.texts = ["Night shifts"]
    sid = first(events(c.post("/v2/lucille/turn", json={"message": "night shifts again"}, headers=H)),
                "session")["sessionId"]
    one = c.get(f"/v2/lucille/threads/{sid}", headers=H).json()
    assert one["title"] == "Night shifts" and [m["role"] for m in one["messages"]] == ["user", "assistant"]
    assert c.get(f"/v2/lucille/threads/{sid}", headers={"X-Test-Uid": "u2"}).status_code == 404
    assert c.delete(f"/v2/lucille/threads/{sid}", headers=H).status_code == 204
    assert c.get("/v2/lucille/threads", headers=H).json()["items"] == []
    assert c.delete(f"/v2/lucille/threads/{sid}", headers=H).status_code == 404


# ── audio fallback ───────────────────────────────────────────────────────────

def test_turn_audio_transcribes_then_streams(lucille, monkeypatch):
    c, fake = lucille
    consent(c)
    import escape_api.routers.voice as voice

    async def fake_whisper(name, data):
        return "I feel wired after my shift"

    monkeypatch.setattr(voice, "_whisper", fake_whisper)
    fake.scripts = [_say("Wired after a shift is hard to put down.")]
    r = c.post("/v2/lucille/turn/audio", files={"file": ("turn.m4a", b"abc", "audio/m4a")},
               data={"sessionId": "", "speak": "false"}, headers=H)
    assert r.status_code == 200
    evs = events(r)
    assert first(evs, "session") and "shift" in text_of(evs)
    assert fake.stream_calls[-1]["messages"][-1]["content"] == "I feel wired after my shift"


def test_turn_audio_requires_voice_consent(lucille):
    c, _ = lucille
    consent(c, wellbeing_data=True, ai_reflection=True, personalization=True)
    r = c.post("/v2/lucille/turn/audio", files={"file": ("turn.m4a", b"abc", "audio/m4a")}, headers=H)
    assert r.status_code == 403 and r.json()["detail"]["purpose"] == "voice"


def test_speak_returns_audio_on_done(lucille):
    from escape_api import tts
    c, fake = lucille
    consent(c)

    async def fake_voice(text):
        return b"ID3mp3", "openai", "nova"

    tts.set_override(fake_voice)
    try:
        fake.scripts = [_say("Breathe with me for a minute.")]
        done = first(events(c.post("/v2/lucille/turn",
                                   json={"message": "talk to me", "mode": "voice", "speak": True},
                                   headers=H)), "done")
        assert done["audioMime"] == "audio/mpeg" and done["audio"]
    finally:
        tts.set_override(None)


def test_truncated_answer_is_reported_not_passed_off_as_finished(lucille):
    c, fake = lucille
    consent(c)
    fake.scripts = [[{"type": "text", "text": "So the first thing I'd"},
                     {"type": "stop", "reason": "max_tokens"}]]
    evs = events(c.post("/v2/lucille/turn", json={"message": "tell me everything"}, headers=H))
    err = first(evs, "error")
    assert err["code"] == "answer_truncated"
    assert types(evs)[-1] == "done"


def test_picks_up_a_thread_the_openai_edition_started(lucille):
    """Memory since inception: history lives in the shared chat_sessions collection."""
    c, fake = lucille
    consent(c)
    c.fakes["firebase"].sessions["legacy_1"] = {
        "session_id": "legacy_1", "user_id": "u1", "summary": "",
        "messages": [{"type": "human", "content": "my sister is moving out"},
                     {"type": "ai", "content": "that's a big change"}]}
    fake.scripts = [_say("You mentioned your sister moving out.")]
    evs = events(c.post("/v2/lucille/turn",
                        json={"message": "it's quiet now", "sessionId": "legacy_1"}, headers=H))
    assert first(evs, "session")["sessionId"] == "legacy_1"
    sent = fake.stream_calls[-1]["messages"]
    assert sent[0]["content"] == "my sister is moving out"
    assert sent[1]["content"] == "that's a big change"
    assert sent[-1]["content"] == "it's quiet now"
