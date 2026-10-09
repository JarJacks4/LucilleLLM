"""/v1/journal — the new scan-first Journal (Free write, Lucille guided, Gratitude, Ritual Spark)."""

from __future__ import annotations

import hashlib
from datetime import date, timedelta
from typing import List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from escape_api import core, journal, mood
from escape_api.deps import Caller, caller, has_consent, require_consent
from escape_api.repo import user_path
from escape_api.settings import cfg

router = APIRouter(prefix="/v1/journal", tags=["v1 · journal"])

Mode = Literal["free", "guided", "gratitude", "ritual"]


class MoodSeed(BaseModel):
    checkinId: Optional[str] = None
    word: Optional[str] = None
    valence: Optional[float] = Field(None, ge=0, le=1)
    energy: Optional[float] = Field(None, ge=0, le=1)
    bpm: Optional[float] = None
    text: Optional[str] = Field(None, max_length=1000)


class VoiceIn(BaseModel):
    transcript: Optional[str] = Field(None, max_length=20000)
    audioPath: Optional[str] = Field(None, description="Firebase Storage path, if the user kept the audio")
    durationSec: Optional[float] = None


class EntryIn(BaseModel):
    mode: Mode = "free"
    status: Literal["draft", "saved"] = "saved"
    title: Optional[str] = Field(None, max_length=140)
    body: Optional[str] = Field(None, max_length=20000)
    gratitude: List[str] = Field(default_factory=list, max_length=3)
    intention: Optional[str] = Field(None, max_length=500)
    prompt: Optional[str] = Field(None, max_length=500)
    moodSeed: Optional[MoodSeed] = None
    tags: List[str] = Field(default_factory=list, max_length=12)
    energyCenter: Optional[Literal["grounding", "creativity", "power", "connection", "expression", "intuition", "purpose"]] = None
    voice: Optional[VoiceIn] = None
    photoPath: Optional[str] = None
    breathFirst: bool = False


class EntryPatch(BaseModel):
    status: Optional[Literal["draft", "saved"]] = None
    title: Optional[str] = Field(None, max_length=140)
    body: Optional[str] = Field(None, max_length=20000)
    gratitude: Optional[List[str]] = Field(None, max_length=3)
    intention: Optional[str] = Field(None, max_length=500)
    tags: Optional[List[str]] = Field(None, max_length=12)
    energyCenter: Optional[str] = None
    photoPath: Optional[str] = None
    userReframe: Optional[str] = Field(None, max_length=1000)
    moodAfter: Optional[str] = Field(None, description="Mood word after reflecting")


class PromptIn(BaseModel):
    mode: Mode = "free"
    word: Optional[str] = None
    personalize: bool = Field(False, description="Guided mode only: let Lucille tailor the prompt (counts toward quota)")


class LetterIn(BaseModel):
    body: str = Field(..., min_length=1, max_length=10000)
    deliverInDays: Literal[7, 30, 90] = 30
    moodWord: Optional[str] = None


# ───────────────────────── helpers ─────────────────────────

def _get_entry(c: Caller, entry_id: str) -> dict:
    e = c.repo.get(user_path(c.uid, "journal_entries", entry_id))
    if not e:
        raise HTTPException(404, "Entry not found")
    return e


def _resolve_mood(c: Caller, seed: Optional[MoodSeed], text: str) -> Optional[dict]:
    if seed and seed.checkinId:
        ck = c.repo.get(user_path(c.uid, "mood_checkins", seed.checkinId))
        if ck:
            return {k: ck.get(k) for k in ("word", "family", "tone", "valence", "energy")} | {"checkinId": ck["id"], "bpm": ck.get("bpm")}
    if seed and (seed.word or seed.valence is not None):
        r = mood.fuse_inputs(word=seed.word, valence=seed.valence, energy=seed.energy, text=seed.text, bpm=seed.bpm)
        return {k: r[k] for k in ("word", "family", "tone", "valence", "energy")} | {"bpm": seed.bpm}
    w = core.interpret_text(text) if text else None
    if w:
        return {"word": w["word"], "family": w["family"], "valence": w["valence"], "energy": w["energy"],
                "tone": core.tone_for(w["valence"], w["energy"], w["family"] == "tender"), "inferred": True}
    return None


def _on_saved(c: Caller, e: dict) -> dict:
    """Streak, coins, and feed the mood orb when an entry is saved (once)."""
    if e.get("savedOnce"):
        return {}
    core.touch_activity(c.repo, c.uid, c.tz, "journal")
    coins = core.award_coins(c.repo, c.uid, c.tz, "journal_entry")
    if e["mode"] == "gratitude" and len([g for g in e.get("gratitude", []) if g.strip()]) >= 3:
        coins += core.award_coins(c.repo, c.uid, c.tz, "gratitude")
    feed = {}
    m = e.get("mood")
    if m and not m.get("checkinId"):       # feeds the scans: becomes a 'journal' check-in
        ck = mood.record_checkin(c.repo, c.uid, c.tz, {**m, "confidence": 0.4, "inputs": ["journal"]}, "journal",
                                 extra={"journalEntryId": e["id"]})
        feed = {"checkinId": ck["id"]}
        m["checkinId"] = ck["id"]
    c.repo.set(user_path(c.uid, "journal_entries", e["id"]), {"savedOnce": True, "mood": m}, merge=True)
    st = core.get_state(c.repo, c.uid)
    return {"coinsAwarded": coins, "streakDays": st["streakDays"], **feed}


# ───────────────────────── home + modes + prompt ─────────────────────────

@router.get("/home", summary="Journal home: one Mood Orb (period), last scan seed, weekly reflection, modes")
async def home(c: Caller = Depends(caller), period: Literal["week", "month", "year"] = "week"):
    st = core.get_state(c.repo, c.uid)
    last = st.get("lastMood")
    fresh = None
    if last:
        age_h = (core.now_utc() - core.parse_iso(last["createdAt"])).total_seconds() / 3600
        if age_h <= 12:
            fresh = {**last, "ageHours": round(age_h, 1)}
    weekly = await journal.weekly_reflection(c.repo, c.uid, c.tz, allow_llm=False, generate=False)
    return {
        "greetingName": st.get("displayName"),
        "lastScan": fresh,
        "orb": mood.summary(c.repo, c.uid, c.tz, period, core.is_night(c.tz)),
        "statsLink": core.resolve_route("MoodStatsV2", {"period": period}),
        "weeklyReflection": weekly,
        "modes": journal.modes_list(),
        "streakDays": st["streakDays"], "coins": st["coins"],
        "night": core.is_night(c.tz),
    }


@router.get("/modes", summary="The four modes with icon + colour keys (static)")
def modes():
    return {"modes": journal.modes_list()}


@router.post("/prompt", summary="A prompt for the chosen mode + mood. Dataset first; guided can be personalised")
async def prompt(body: PromptIn, c: Caller = Depends(caller)):
    word = body.word or (core.get_state(c.repo, c.uid).get("lastMood") or {}).get("word")
    p = journal.dataset_prompt(body.mode, word, seed=int(hashlib.sha256(f"{c.uid}{core.local_day(c.tz)}{body.mode}".encode()).hexdigest()[:8], 16))
    if (body.mode == "guided" and body.personalize and has_consent(c, "ai_reflection")
            and core.check_quota(c.repo, c.uid, c.tz, "guided_prompt", cfg().quota_guided_prompt_per_day)):
        recent = [d for _, d in c.repo.query(user_path(c.uid, "journal_entries"), order_by="createdAt", desc=True, limit=3)]
        themes = sorted({t for d in recent for t in (d.get("themes") or [])})[:4]
        from escape_api import llm
        out = await llm.complete_json(
            "Write ONE journaling question (max 22 words) for someone feeling the given mood, gently touching "
            "one of their recent themes if natural. Key: prompt.",
            f"Mood: {word or 'mixed'}\nRecent themes: {', '.join(themes) or 'none'}",
            {"prompt": p["prompt"]},
            cache_key=f"guided:{(word or 'mixed').lower()}:{','.join(themes)}",   # cohort cache
            max_tokens=80)
        p.update(prompt=out["prompt"], source=out["_source"], aiGenerated=out["_source"] in ("llm", "cache"))
    return p


# ───────────────────────── entries CRUD ─────────────────────────

@router.post("/entries", status_code=201, summary="Create an entry (draft or saved). Safety-checked; feeds the Mood Orb")
def create_entry(body: EntryIn, c: Caller = Depends(caller)):
    require_consent(c, "wellbeing_data")
    if body.voice and body.voice.transcript and not has_consent(c, "voice"):
        raise HTTPException(403, {"code": "consent_required", "purpose": "voice"})
    now = core.now_utc()
    e = {
        "id": core.new_id("je"), "mode": body.mode, "status": body.status,
        "title": body.title, "body": body.body or "",
        "gratitude": [g.strip() for g in body.gratitude if g and g.strip()][:3],
        "intention": body.intention, "prompt": body.prompt, "tags": body.tags[:12],
        "energyCenter": body.energyCenter, "voice": body.voice.model_dump() if body.voice else None,
        "photoPath": body.photoPath, "breathFirst": body.breathFirst,
        "createdAt": core.iso(now), "updatedAt": core.iso(now),
        "localDay": core.local_day(c.tz, now), "localHour": now.astimezone(core.safe_tz(c.tz)).hour,
    }
    text = journal.entry_text(e)
    e["wordCount"] = len(text.split())
    e["themes"] = journal.extract_themes(text)
    e["mood"] = _resolve_mood(c, body.moodSeed, text)
    safety = core.safety_check(text)
    e["flagged"] = safety["crisis"]
    c.repo.set(user_path(c.uid, "journal_entries", e["id"]), e)
    effects = _on_saved(c, e) if body.status == "saved" else {}
    if safety["crisis"]:
        try:
            from models import RiskLevel, SafetyEventType
            from safety_service import get_safety_service  # existing audit log (safety_audit/{uid}/events)
            get_safety_service().log_safety_event(
                user_id=c.uid, session_id=f"journal:{e['id']}", event_type=SafetyEventType.CRISIS_DETECTED,
                risk_level=RiskLevel(safety["riskLevel"]), message_snippet="[journal entry]",
                flags=safety.get("flags", []), action_taken="crisis_resources_shown")
        except Exception:
            pass
    return {"entry": c.repo.get(user_path(c.uid, "journal_entries", e["id"])), "safety": safety, **effects}


@router.get("/entries", summary="List entries, newest first. Filter by mode, theme, day range")
def list_entries(c: Caller = Depends(caller), limit: int = Query(20, ge=1, le=50), cursor: Optional[str] = None,
                 mode: Optional[Mode] = None, theme: Optional[str] = None,
                 status: Literal["saved", "draft", "all"] = "saved"):
    where = []
    if mode:
        where.append(("mode", "==", mode))
    if status != "all":
        where.append(("status", "==", status))
    if theme:
        where.append(("themes", "array_contains", theme))
    rows = c.repo.query(user_path(c.uid, "journal_entries"), where=where, order_by="createdAt", desc=True,
                        limit=limit, start_after=cursor)
    items = [{k: d.get(k) for k in ("id", "mode", "status", "title", "createdAt", "localDay", "themes", "mood",
                                     "wordCount", "energyCenter", "flagged")}
             | {"preview": journal.entry_text(d)[:160], "hasReflection": bool(d.get("reflection"))} for _, d in rows]
    return {"items": items, "nextCursor": items[-1]["createdAt"] if len(items) == limit else None}


@router.get("/entries/{entry_id}", summary="One entry with its stored reflection")
def get_entry(entry_id: str, c: Caller = Depends(caller)):
    return _get_entry(c, entry_id)


@router.patch("/entries/{entry_id}", summary="Edit an entry, save a draft, save the user's reframe or mood after")
def patch_entry(entry_id: str, body: EntryPatch, c: Caller = Depends(caller)):
    e = _get_entry(c, entry_id)
    upd = {k: v for k, v in body.model_dump(exclude_none=True).items() if k != "moodAfter"}
    if body.moodAfter:
        w = core.find_word(body.moodAfter)
        if w:
            upd["moodAfter"] = {"word": w["word"], "family": w["family"], "valence": w["valence"], "energy": w["energy"]}
    e.update(upd)
    text = journal.entry_text(e)
    e["wordCount"] = len(text.split())
    e["themes"] = journal.extract_themes(text) or e.get("themes", [])
    e["updatedAt"] = core.iso(core.now_utc())
    safety = core.safety_check(text)
    e["flagged"] = safety["crisis"]
    if any(k in upd for k in ("body", "gratitude", "intention", "title")):
        e.pop("reflection", None)          # content changed -> old reflection no longer matches
        c.repo.set(user_path(c.uid, "journal_entries", entry_id), e)
    else:
        c.repo.set(user_path(c.uid, "journal_entries", entry_id), e, merge=True)
    effects = _on_saved(c, e) if e.get("status") == "saved" else {}
    return {"entry": c.repo.get(user_path(c.uid, "journal_entries", entry_id)), "safety": safety, **effects}


@router.delete("/entries/{entry_id}", status_code=204, summary="Delete an entry (and the check-in it created)")
def delete_entry(entry_id: str, c: Caller = Depends(caller)):
    e = _get_entry(c, entry_id)
    m = e.get("mood") or {}
    ck = c.repo.get(user_path(c.uid, "mood_checkins", m["checkinId"])) if m.get("checkinId") else None
    if ck and ck.get("source") == "journal":
        mood.delete_checkin(c.repo, c.uid, ck["id"])
    c.repo.delete(user_path(c.uid, "journal_entries", entry_id))


# ───────────────────────── Lucille reflects ─────────────────────────

@router.post("/entries/{entry_id}/reflect", summary="Lucille reflects + optional reframe + tuned soundscape (stored; repeat calls are free)")
async def reflect(entry_id: str, c: Caller = Depends(caller), regenerate: bool = False):
    e = _get_entry(c, entry_id)
    m = e.get("mood") or {"word": "Mixed", "family": "mixed", "tone": "mixed"}
    if e.get("flagged"):
        safety = core.safety_check(journal.entry_text(e))
        return {"reflection": {
            "reflection": "Thank you for trusting this page with something this heavy. You don't have to hold it alone. "
                          "Please reach out to someone right now: you can call or text 988 any time in the US, or your local emergency number.",
            "reframe": None, "themes": e.get("themes", []), "aiGenerated": False, "crisis": True},
            "safety": safety, "suggestions": [], "soundscape": None, "disclosure": core.AI_DISCLOSURE}
    if e.get("reflection") and not regenerate:
        refl = e["reflection"]
    else:
        allow = (has_consent(c, "ai_reflection")
                 and core.check_quota(c.repo, c.uid, c.tz, "reflect", cfg().quota_reflect_per_day))
        out = await journal.reflect(e, m["word"], m["family"], allow_llm=allow)
        refl = {"reflection": out["reflection"], "reframe": out.get("reframe"), "themes": out.get("themes", []),
                "energyCenter": out.get("energyCenter"), "soundscapeCategory": out["soundscapeCategory"],
                "aiGenerated": out["_source"] == "llm", "createdAt": core.iso(core.now_utc())}
        c.repo.set(user_path(c.uid, "journal_entries", entry_id),
                   {"reflection": refl, "themes": refl["themes"] or e.get("themes", [])}, merge=True)
        if out["_source"] == "llm":
            core.award_coins(c.repo, c.uid, c.tz, "journal_reflect")
    return {
        "reflection": refl,
        "mood": m,
        "orb": core.orb_assets(m.get("tone", "mixed"), core.is_night(c.tz)),
        "soundscape": journal.pick_track(refl.get("soundscapeCategory") or "nature", m["word"]),
        "suggestions": mood.suggestions_for(m["word"], c.tz, 2, exclude_kinds={"journal"}),
        "coinsPreview": core.COIN_RULES["journal_entry"][0],
        "disclosure": core.AI_DISCLOSURE,
    }


# ───────────────────────── insights, weekly, resurfacing ─────────────────────────

@router.get("/insights", summary="14-day mood line, top themes, what feeds the Energy Scan")
def insights(c: Caller = Depends(caller), days: int = Query(14, ge=7, le=90)):
    start = (date.fromisoformat(core.local_day(c.tz)) - timedelta(days=days - 1)).isoformat()
    rows = [d for _, d in c.repo.query(user_path(c.uid, "journal_entries"), where=[("localDay", ">=", start)])
            if d.get("status") == "saved"]
    from collections import Counter
    themes = Counter(t for r in rows for t in (r.get("themes") or []))
    centers = Counter(r.get("energyCenter") or journal.ENERGY_CENTER_FOR_THEME.get((r.get("themes") or [None])[0])
                      for r in rows)
    centers.pop(None, None)
    days_map = {d["day"]: d for _, d in c.repo.query(user_path(c.uid, "mood_days"), where=[("day", ">=", start)])}
    spark = []
    for i in range(days):
        day = (date.fromisoformat(start) + timedelta(days=i)).isoformat()
        r = days_map.get(day, {})
        v = (r["sumV"] / r["count"]) if r.get("count") else None
        h = round(v * 100) if v is not None else 0
        spark.append({"day": day, "h": h, "empty": v is None,
                      "c": "#39D9C1" if h > 55 else ("#8E7CD9" if h and h < 40 else "#4CF6F6")})
    active = len({r["localDay"] for r in rows})
    return {
        "days": days, "entries": len(rows), "activeDays": active,
        "rhythm": "on track" if active >= days / 3 else "gentle nudge",
        "spark": spark,
        "themes": [{"theme": t, "count": n} for t, n in themes.most_common(6)],
        "energyCenters": [{"center": k, "count": n} for k, n in centers.most_common()],
        "feedsEnergyScan": True,
    }


@router.get("/weekly", summary="Lucille's weekly reflection (generated once per week, then free)")
async def weekly(c: Caller = Depends(caller)):
    doc = await journal.weekly_reflection(c.repo, c.uid, c.tz, allow_llm=has_consent(c, "ai_reflection"))
    return {"weekly": doc, "disclosure": core.AI_DISCLOSURE if doc and doc.get("aiGenerated") else None}


@router.get("/resurface", summary="'On this day': an entry from ~1 month, 3 months or 1 year ago")
def resurface(c: Caller = Depends(caller)):
    today = date.fromisoformat(core.local_day(c.tz))
    for back in (365, 90, 30):
        lo = (today - timedelta(days=back + 3)).isoformat()
        hi = (today - timedelta(days=back - 3)).isoformat()
        rows = [d for _, d in c.repo.query(user_path(c.uid, "journal_entries"),
                                           where=[("localDay", ">=", lo), ("localDay", "<=", hi)], limit=5)
                if d.get("status") == "saved" and not d.get("flagged")]
        if rows:
            d = rows[0]
            return {"entry": {k: d.get(k) for k in ("id", "mode", "title", "localDay", "mood", "themes")}
                    | {"preview": journal.entry_text(d)[:200]},
                    "daysAgo": back, "question": "How does this land now?"}
    return {"entry": None}


# ───────────────────────── letters to future self ─────────────────────────

@router.post("/letters", status_code=201, summary="Seal a letter to future you (7 / 30 / 90 days)")
def create_letter(body: LetterIn, c: Caller = Depends(caller)):
    require_consent(c, "wellbeing_data")
    now = core.now_utc()
    st = core.get_state(c.repo, c.uid)
    word = body.moodWord or (st.get("lastMood") or {}).get("word")
    doc = {"id": core.new_id("lt"), "body": body.body, "moodWord": word, "createdAt": core.iso(now),
           "deliverAt": core.iso(now + timedelta(days=body.deliverInDays)), "deliverInDays": body.deliverInDays,
           "delivered": False, "flagged": core.safety_check(body.body)["crisis"]}
    c.repo.set(user_path(c.uid, "journal_letters", doc["id"]), doc)
    coins = core.award_coins(c.repo, c.uid, c.tz, "letter")
    core.touch_activity(c.repo, c.uid, c.tz, "journal")
    return {"letter": {k: v for k, v in doc.items() if k != "body"} | {"sealed": True}, "coinsAwarded": coins}


@router.get("/letters", summary="Your letters. Bodies stay sealed until their delivery date")
def list_letters(c: Caller = Depends(caller)):
    now = core.iso(core.now_utc())
    out = []
    for _, d in c.repo.query(user_path(c.uid, "journal_letters"), order_by="deliverAt"):
        due = d["deliverAt"] <= now
        out.append({k: v for k, v in d.items() if k != "body" or due} | {"sealed": not due})
    return {"items": out}


@router.delete("/letters/{letter_id}", status_code=204)
def delete_letter(letter_id: str, c: Caller = Depends(caller)):
    c.repo.delete(user_path(c.uid, "journal_letters", letter_id))
