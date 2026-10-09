"""/v1/mood — Mood Scan results, check-ins, the conglomerate Mood Orb, Mood Stats."""

from __future__ import annotations

from typing import List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from escape_api import core, mood
from escape_api.deps import Caller, caller, has_consent, require_consent
from escape_api.repo import user_path

router = APIRouter(prefix="/v1/mood", tags=["v1 · mood"])


class PulseIn(BaseModel):
    bpm: Optional[float] = Field(None, ge=30, le=220)
    hrv: Optional[float] = Field(None, ge=0, le=400)
    quality: Optional[float] = Field(None, ge=0, le=1, description="rPPG signal quality 0-1")


class CheckinIn(BaseModel):
    source: Literal["scan", "tap", "voice", "text", "journal", "session_before", "session_after",
                    "energy_scan", "checkin", "healthkit"] = "scan"
    word: Optional[str] = Field(None, description="Mood word the user tapped, e.g. 'Restless'")
    valence: Optional[float] = Field(None, ge=0, le=1, description="Mood Field x: unpleasant 0 -> pleasant 1")
    energy: Optional[float] = Field(None, ge=0, le=1, description="Mood Field y: calm 0 -> charged 1")
    text: Optional[str] = Field(None, max_length=2000, description="Voice transcript or typed words")
    pulse: Optional[PulseIn] = None
    tags: List[str] = Field(default_factory=list, max_length=12)
    note: Optional[str] = Field(None, max_length=1000)
    journalEntryId: Optional[str] = None
    sessionId: Optional[str] = None
    lat: Optional[float] = Field(None, ge=-90, le=90, description="Only to pick day/night orb; not stored")
    lon: Optional[float] = Field(None, ge=-180, le=180)


class InterpretIn(BaseModel):
    text: Optional[str] = Field(None, max_length=2000)
    word: Optional[str] = None
    valence: Optional[float] = Field(None, ge=0, le=1)
    energy: Optional[float] = Field(None, ge=0, le=1)
    pulse: Optional[PulseIn] = None


def _reading(body) -> dict:
    p = body.pulse or PulseIn()
    return mood.fuse_inputs(word=body.word, valence=body.valence, energy=body.energy,
                            text=body.text, bpm=p.bpm, hrv=p.hrv)


@router.get("/words", summary="Mood vocabulary with Mood Field coordinates (static, cache 24h)")
def words():
    d = core.dataset("mood_words")
    return {"words": [{k: w[k] for k in ("word", "valence", "energy", "family")} for w in d["words"]],
            "families": d["families"], "tones": d["tones"]}


@router.post("/interpret", summary="Preview a reading from scan inputs without saving (free, no LLM)")
def interpret(body: InterpretIn, c: Caller = Depends(caller)):
    r = _reading(body)
    return {**r, "orb": core.orb_assets(r["tone"], core.is_night(c.tz)),
            "suggestions": mood.suggestions_for(r["word"], c.tz)}


@router.post("/checkins", status_code=201, summary="Save a Mood Scan / check-in. Returns the orb + suggestions")
def create_checkin(body: CheckinIn, c: Caller = Depends(caller)):
    require_consent(c, "wellbeing_data")
    if body.pulse and body.pulse.bpm and not has_consent(c, "camera_scan") and body.source == "scan":
        raise HTTPException(403, {"code": "consent_required", "purpose": "camera_scan"})
    r = _reading(body)
    safety = core.safety_check(" ".join(filter(None, [body.text, body.note])))
    doc = mood.record_checkin(c.repo, c.uid, c.tz, r, body.source, extra={
        "bpm": body.pulse.bpm if body.pulse else None,
        "hrv": body.pulse.hrv if body.pulse else None,
        "tags": body.tags[:12], "note": body.note,
        "journalEntryId": body.journalEntryId, "sessionId": body.sessionId,
        "flagged": safety["crisis"] or None,
    })
    st = core.touch_activity(c.repo, c.uid, c.tz, "mood")
    coins = core.award_coins(c.repo, c.uid, c.tz, "energy_scan" if body.source == "energy_scan" else "mood_checkin")
    night = core.is_night(c.tz, body.lat, body.lon)
    return {
        "checkin": doc,
        "lucille": {"word": doc["word"], "line": _lucille_line(doc["word"], doc["family"]), "aiGenerated": False},
        "orb": core.orb_assets(doc["tone"], night),
        "suggestions": [] if safety["crisis"] else mood.suggestions_for(doc["word"], c.tz),
        "journalSeed": {"checkinId": doc["id"], "word": doc["word"], "tone": doc["tone"],
                        "bpm": doc.get("bpm"), "text": body.text,
                        "deeplink": core.resolve_route("JournalHomeV2", {"seed": doc["id"]})},
        "safety": safety,
        "streakDays": st["streakDays"], "coinsAwarded": coins,
    }


def _lucille_line(word: str, family: str) -> str:
    lines = {
        "energized": "There's real energy in you right now. Let's put it somewhere good.",
        "calm": "You're in a settled place. Worth noticing what got you here.",
        "tender": "Something soft and warm is close to the surface. Let it be there.",
        "mixed": "A bit of everything today, and that's allowed.",
        "restless": "Your system is running fast. We can slow it down together.",
        "heavy": "It's heavy right now. You don't have to fix it all at once.",
    }
    return f"{word}. " + lines.get(family, lines["mixed"])


@router.get("/checkins", summary="List check-ins, newest first (cursor = createdAt of last item)")
def list_checkins(c: Caller = Depends(caller), limit: int = Query(30, ge=1, le=100),
                  cursor: Optional[str] = None, day_from: Optional[str] = None, day_to: Optional[str] = None):
    where = []   # filter on the ordered field so Firestore needs no composite index
    if day_from:
        where.append(("createdAt", ">=", day_from))
    if day_to:
        where.append(("createdAt", "<=", day_to + "T23:59:59Z"))
    rows = c.repo.query(user_path(c.uid, "mood_checkins"), where=where, order_by="createdAt", desc=True,
                        limit=limit, start_after=cursor)
    items = [d for _, d in rows]
    return {"items": items, "nextCursor": items[-1]["createdAt"] if len(items) == limit else None}


@router.get("/latest", summary="Most recent reading (seeds 'Save this moment' in the Journal)")
def latest(c: Caller = Depends(caller)):
    st = c.repo.get(user_path(c.uid, "state")) or {}
    last = st.get("lastMood")
    if not last:
        return {"latest": None}
    return {"latest": last, "orb": core.orb_assets(last["tone"], core.is_night(c.tz))}


@router.delete("/checkins/{checkin_id}", status_code=204, summary="Delete one check-in (rollups adjust)")
def delete_checkin(checkin_id: str, c: Caller = Depends(caller)):
    if not mood.delete_checkin(c.repo, c.uid, checkin_id):
        raise HTTPException(404, "Not found")


@router.get("/summary", summary="The single conglomerate Mood Orb for week / month / year")
def get_summary(c: Caller = Depends(caller), period: Literal["week", "month", "year"] = "week",
                lat: Optional[float] = None, lon: Optional[float] = None):
    return mood.summary(c.repo, c.uid, c.tz, period, core.is_night(c.tz, lat, lon))


@router.get("/stats", summary="Mood Stats page: all three periods + insights in one call")
def stats(c: Caller = Depends(caller), period: Literal["week", "month", "year"] = "week"):
    night = core.is_night(c.tz)
    main = mood.summary(c.repo, c.uid, c.tz, period, night)
    others = {p: {k: mood.summary(c.repo, c.uid, c.tz, p, night)[k] for k in ("word", "tone", "size", "entries")}
              for p in ("week", "month", "year") if p != period}
    return {"summary": main, "periods": others,
            "insightsLink": core.resolve_route("JournalHomeV2", {"screen": "insights"})}


@router.get("/suggestions", summary="Things to do in Escape to help with a mood (allowlisted deep links)")
def suggestions(c: Caller = Depends(caller), word: Optional[str] = None, limit: int = Query(3, ge=1, le=6)):
    if not word:
        st = c.repo.get(user_path(c.uid, "state")) or {}
        word = (st.get("lastMood") or {}).get("word", "Neutral")
    return {"word": word, "items": mood.suggestions_for(word, c.tz, limit)}


@router.get("/orb", summary="Which Mood Orb loop + background to show right now")
def orb(c: Caller = Depends(caller), tone: Optional[str] = None, lat: Optional[float] = None, lon: Optional[float] = None):
    if tone not in core.dataset("mood_words")["tones"]:
        st = c.repo.get(user_path(c.uid, "state")) or {}
        tone = (st.get("lastMood") or {}).get("tone", "mixed")
    return core.orb_assets(tone, core.is_night(c.tz, lat, lon))
