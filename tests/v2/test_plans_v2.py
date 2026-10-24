"""/v2/plans — Claude authoring, catalog fallback, CRUD, documents and charts."""

import json
from datetime import date, timedelta

import pytest

from escape_api import core, documents
from escape_api.claude import client as claude
from escape_api.repo import user_path

from tests.v2.conftest import H, FakeClaude, consent


def _claude_plan_json(days=3):
    return json.dumps({
        "title": "Your quiet week",
        "why": "Your last three check-ins were restless, so this leans on wind-down.",
        "days": [{"day": d, "items": [{"id": "breath_box", "why": "it slows a racing chest"}]}
                 for d in range(1, days + 1)],
    })


# ── creating ─────────────────────────────────────────────────────────────────

def test_plans_require_personalization_consent(client):
    consent(client, wellbeing_data=True, ai_reflection=True)
    r = client.post("/v2/plans", json={"goals": ["less_stress"]}, headers=H)
    assert r.status_code == 403 and r.json()["detail"]["purpose"] == "personalization"


def test_claude_writes_the_plan_with_a_why_per_item(client):
    consent(client)
    claude.set_client(FakeClaude(texts=[_claude_plan_json(3)]))
    r = client.post("/v2/plans", json={"goals": ["less_stress"], "minutesPerDay": 10, "days": 3},
                    headers=H)
    assert r.status_code == 201, r.text
    plan = r.json()["plan"]
    assert plan["provider"] == "claude" and plan["title"] == "Your quiet week"
    assert "restless" in plan["why"] and plan["days"] == 3 and plan["revision"] == 1
    assert plan["archived"] is False
    assert len(plan["items"]) == 3
    for i in plan["items"]:
        assert i["why"] and i["route"] and i["deeplink"]["url"].startswith("escape://")
        assert set(("id", "day", "title", "kind", "minutes", "route", "params", "why", "done", "doneAt")) <= set(i)
    assert any(s["label"].startswith("Drafted") for s in r.json()["statuses"])


def test_falls_back_to_the_catalog_without_claude(client):
    consent(client)
    claude.set_client(None)
    r = client.post("/v2/plans", json={"goals": ["sleep_better"], "days": 7}, headers=H)
    assert r.status_code == 201
    plan = r.json()["plan"]
    assert plan["provider"] == "catalog" and r.json()["claudeAvailable"] is False
    assert len(plan["items"]) >= 7 and plan["why"]
    acts = {a["id"] for a in client.get("/v1/selfcare/activities").json()["items"]}
    assert all(i["activityId"] in acts for i in plan["items"])


def test_falls_back_when_claude_returns_rubbish(client):
    consent(client)
    claude.set_client(FakeClaude(texts=['{"days": [{"day": 1, "items": [{"id": "not_a_real_activity"}]}]}']))
    plan = client.post("/v2/plans", json={"goals": ["focus"], "days": 3}, headers=H).json()["plan"]
    assert plan["provider"] == "catalog" and plan["items"]


def test_claude_plan_quota_falls_back(client, monkeypatch):
    from escape_api.settings import reload_settings
    consent(client)
    monkeypatch.setenv("QUOTA_CLAUDE_PLAN_PER_DAY", "1")
    reload_settings()
    claude.set_client(FakeClaude(texts=[_claude_plan_json(), _claude_plan_json()]))
    first = client.post("/v2/plans", json={"goals": ["focus"], "days": 3}, headers=H).json()["plan"]
    second = client.post("/v2/plans", json={"goals": ["focus"], "days": 3}, headers=H).json()["plan"]
    assert first["provider"] == "claude" and second["provider"] == "catalog"


def test_creating_a_plan_archives_the_previous_one(client):
    consent(client)
    claude.set_client(None)
    a = client.post("/v2/plans", json={"goals": ["focus"], "days": 3}, headers=H).json()["plan"]
    b = client.post("/v2/plans", json={"goals": ["move_more"], "days": 3}, headers=H).json()["plan"]
    assert client.get("/v2/plans/active", headers=H).json()["plan"]["id"] == b["id"]
    assert client.get(f"/v2/plans/{a['id']}", headers=H).json()["plan"]["archived"] is True


# ── reading, completing, adjusting, archiving ────────────────────────────────

def _plan(client):
    consent(client)
    claude.set_client(None)
    return client.post("/v2/plans", json={"goals": ["less_stress"], "days": 7, "minutesPerDay": 20},
                       headers=H).json()["plan"]


def test_active_today_and_complete_awards_coins_and_score(client):
    plan = _plan(client)
    active = client.get("/v2/plans/active", headers=H).json()
    assert active["plan"]["id"] == plan["id"] and active["today"]
    item = active["today"][0]
    r = client.post(f"/v2/plans/{plan['id']}/items/{item['id']}/complete", headers=H)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["coins"] == 10 and 0 <= j["score"]["score"] <= 100
    assert next(i for i in j["plan"]["items"] if i["id"] == item["id"])["done"] is True
    # idempotent: no second payout
    again = client.post(f"/v2/plans/{plan['id']}/items/{item['id']}/complete", headers=H).json()
    assert again["coins"] == 0
    assert client.post(f"/v2/plans/{plan['id']}/items/nope/complete", headers=H).status_code == 404


def test_patch_easier_harder_and_shift_days(client):
    plan = _plan(client)
    before = len(plan["items"])
    easier = client.patch(f"/v2/plans/{plan['id']}", json={"adjust": "easier", "note": "too much"},
                          headers=H).json()
    assert len(easier["plan"]["items"]) < before and easier["changed"]
    assert easier["plan"]["minutesPerDay"] == 15 and easier["plan"]["revision"] == 2
    harder = client.patch(f"/v2/plans/{plan['id']}", json={"adjust": "harder"}, headers=H).json()
    assert len(harder["plan"]["items"]) > len(easier["plan"]["items"])
    # push the start into the past, then shift
    stale = harder["plan"]
    yesterday = (date.fromisoformat(core.local_day("America/New_York")) - timedelta(days=2)).isoformat()
    for i in stale["items"]:
        i["date"] = yesterday
    client.repo.set(user_path("u1", "plans_v2", stale["id"]), stale)
    shifted = client.patch(f"/v2/plans/{stale['id']}", json={"adjust": "shift_days"}, headers=H).json()
    assert "forward by 2 days" in shifted["changed"][0]
    assert all(i["date"] >= core.local_day("America/New_York") for i in shifted["plan"]["items"])
    assert client.patch(f"/v2/plans/{stale['id']}", json={"adjust": "sideways"},
                        headers=H).status_code == 422


def test_delete_archives_and_never_hard_deletes(client):
    plan = _plan(client)
    r = client.delete(f"/v2/plans/{plan['id']}", headers=H)
    assert r.status_code == 200 and r.json()["archived"] is True
    assert client.get(f"/v2/plans/{plan['id']}", headers=H).json()["plan"]["archived"] is True
    assert client.get("/v2/plans/active", headers=H).json()["plan"] is None


def test_plans_are_per_user(client):
    plan = _plan(client)
    assert client.get(f"/v2/plans/{plan['id']}", headers={"X-Test-Uid": "u2"}).status_code == 404


# ── charts ───────────────────────────────────────────────────────────────────

def test_chart_empty_state_is_honest(client):
    plan = _plan(client)
    j = client.get(f"/v2/plans/{plan['id']}/chart?metric=mood&range=7d", headers=H).json()
    assert j["series"]["empty"] is True
    assert "Not enough history" in j["series"]["note"]
    assert "Not enough history" in j["svg"] and j["svg"].startswith("<svg")
    assert all(p["value"] is None for p in j["series"]["points"])
    assert j["palette"]["night"] == "#0B1230"


def test_chart_uses_real_data(client):
    plan = _plan(client)
    for word in ("Calm", "Sad"):
        client.post("/v1/mood/checkins", json={"word": word}, headers=H)
    item = client.get("/v2/plans/active", headers=H).json()["today"][0]
    client.post(f"/v2/plans/{plan['id']}/items/{item['id']}/complete", headers=H)

    mood = client.get(f"/v2/plans/{plan['id']}/chart?metric=mood&range=7d", headers=H).json()["series"]
    today = core.local_day("America/New_York")
    point = next(p for p in mood["points"] if p["day"] == today)
    assert point["value"] is not None and 0 <= point["value"] <= 100

    adherence = client.get(f"/v2/plans/{plan['id']}/chart?metric=adherence&range=7d",
                           headers=H).json()["series"]
    todays = next(p for p in adherence["points"] if p["day"] == today)
    assert todays["value"] == round(100 / len(client.get("/v2/plans/active", headers=H).json()["today"]))
    assert "<rect" in documents.chart_svg(adherence)

    minutes = client.get(f"/v2/plans/{plan['id']}/chart?metric=minutes&range=7d", headers=H).json()["series"]
    assert next(p for p in minutes["points"] if p["day"] == today)["value"] == item["minutes"]

    streak = client.get(f"/v2/plans/{plan['id']}/chart?metric=streak&range=7d", headers=H).json()["series"]
    assert next(p for p in streak["points"] if p["day"] == today)["value"] >= 1


def test_chart_rejects_unknown_metric(client):
    plan = _plan(client)
    assert client.get(f"/v2/plans/{plan['id']}/chart?metric=vibes", headers=H).status_code == 422


def test_score_chart_reads_the_daily_snapshots(client):
    _plan(client)
    client.get("/v1/selfcare/score", headers=H)                 # writes today's snapshot
    data = documents.series(client.repo, "u1", "America/New_York", "score", "7d")
    today = core.local_day("America/New_York")
    assert next(p for p in data["points"] if p["day"] == today)["value"] is not None


# ── documents ────────────────────────────────────────────────────────────────

def test_pdf_document_is_named_and_served_without_a_bucket(client):
    pytest.importorskip("reportlab", reason="reportlab is not installed on this machine")
    plan = _plan(client)
    r = client.post(f"/v2/plans/{plan['id']}/document",
                    json={"format": "pdf", "include": ["schedule", "progress"]}, headers=H)
    assert r.status_code == 200, r.text
    j = r.json()
    today = core.local_day("America/New_York")
    assert j["filename"] == f"Escape-Plan-Your-feel-less-stressed-week-{today}.pdf"
    assert j["mime"] == "application/pdf" and j["bytes"] > 1000
    assert j["storage"] == "api" and j["url"].endswith(f"/v2/plans/{plan['id']}/documents/{j['id']}")
    assert j["expiresAt"] > core.iso(core.now_utc())
    assert [c["metric"] for c in j["charts"]] == ["adherence", "mood"]

    dl = client.get(f"/v2/plans/{plan['id']}/documents/{j['id']}", headers=H)
    assert dl.status_code == 200 and dl.content[:5] == b"%PDF-"
    assert j["filename"] in dl.headers["content-disposition"]
    # someone else's token can't fetch it
    assert client.get(f"/v2/plans/{plan['id']}/documents/{j['id']}",
                      headers={"X-Test-Uid": "u2"}).status_code == 404


def test_markdown_document_contains_the_real_schedule_and_a_disclaimer(client):
    plan = _plan(client)
    j = client.post(f"/v2/plans/{plan['id']}/document", json={"format": "md"}, headers=H).json()
    body = client.get(f"/v2/plans/{plan['id']}/documents/{j['id']}", headers=H).text
    assert body.startswith("# ") and "## Schedule" in body and "Day 1" in body
    assert plan["items"][0]["title"] in body
    assert "not a therapist" in body and "## Progress" in body


def test_docx_document_or_a_clear_503(client):
    plan = _plan(client)
    r = client.post(f"/v2/plans/{plan['id']}/document", json={"format": "docx"}, headers=H)
    if r.status_code == 503:
        assert r.json()["detail"]["code"] == "docx_unavailable"
        pytest.skip("python-docx is not installed on this machine")
    assert r.status_code == 200
    assert r.json()["mime"].endswith("wordprocessingml.document")
    data = client.get(f"/v2/plans/{plan['id']}/documents/{r.json()['id']}", headers=H).content
    assert data[:2] == b"PK"


def test_document_quota(client, monkeypatch):
    from escape_api.settings import reload_settings
    plan = _plan(client)
    monkeypatch.setenv("QUOTA_CLAUDE_DOC_PER_DAY", "1")
    reload_settings()
    assert client.post(f"/v2/plans/{plan['id']}/document", json={"format": "md"},
                       headers=H).status_code == 200
    r = client.post(f"/v2/plans/{plan['id']}/document", json={"format": "md"}, headers=H)
    assert r.status_code == 429 and r.json()["detail"]["code"] == "quota_exceeded"


def test_expired_documents_are_not_served(client):
    plan = _plan(client)
    j = client.post(f"/v2/plans/{plan['id']}/document", json={"format": "md"}, headers=H).json()
    client.repo.set(user_path("u1", documents.DOCS, j["id"]),
                    {"expiresAt": "2000-01-01T00:00:00Z"}, merge=True)
    r = client.get(f"/v2/plans/{plan['id']}/documents/{j['id']}", headers=H)
    assert r.status_code == 404 and r.json()["detail"]["code"] == "document_expired"


def test_document_filename_slug_is_safe():
    plan = {"title": "Your “quiet” week / v2", "items": []}
    name = documents.document_filename(plan, "pdf", "UTC")
    assert name.startswith("Escape-Plan-Your-quiet-week-v2-") and name.endswith(".pdf")
    assert "/" not in name and " " not in name


def test_empty_series_reports_no_numbers_at_all(client):
    plan = _plan(client)
    mood = documents.series(client.repo, "u1", "America/New_York", "mood", "30d", plan)
    assert mood["empty"] is True and mood["average"] is None and mood["max"] == 0
    # one scheduled plan day is already worth showing, so adherence is not "empty"
    item = client.get("/v2/plans/active", headers=H).json()["today"][0]
    client.post(f"/v2/plans/{plan['id']}/items/{item['id']}/complete", headers=H)
    plan = client.get(f"/v2/plans/{plan['id']}", headers=H).json()["plan"]
    adherence = documents.series(client.repo, "u1", "America/New_York", "adherence", "30d", plan)
    assert adherence["empty"] is False and adherence["average"] is not None
