"""/v1/me, /v1/home, /v1/suggestions — the cross-app pieces (streak, coins, energy centers, one-call Home)."""

from __future__ import annotations

from typing import Literal, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from escape_api import core, mood, selfcare
from escape_api.deps import Caller, caller
from escape_api.repo import user_path

router = APIRouter(prefix="/v1", tags=["v1 · me & home"])


class PrefsIn(BaseModel):
    timezone: Optional[str] = Field(None, max_length=64, description="IANA, e.g. America/New_York")
    displayName: Optional[str] = Field(None, max_length=60)
    wakeTime: Optional[str] = Field(None, pattern=r"^\d{2}:\d{2}$")
    birthYear: Optional[int] = Field(None, ge=1900, le=2030, description="For minor protections; stored as an age band only")
    blend: Optional[bool] = None
    lucilleWhisper: Optional[bool] = None
    defaultBrainwave: Optional[Literal["delta", "theta", "alpha", "beta"]] = None
    isPremium: Optional[bool] = Field(None, description="Ignored from clients; set by the RevenueCat webhook")


def _public_state(st: dict) -> dict:
    keep = ("streakDays", "bestStreak", "coins", "lifetimeCoins", "timezone", "displayName", "wakeTime",
            "lastMood", "lastActiveDay", "prefs", "ageBand", "isMinor", "isPremium")
    return {k: st.get(k) for k in keep}


@router.get("/me", summary="Profile state: streak, coins, energy level, prefs")
def me(c: Caller = Depends(caller)):
    st = core.get_state(c.repo, c.uid)
    return {"uid": c.uid, **_public_state(st), "energy": core.energy_progress(int(st.get("lifetimeCoins", 0)))}


@router.patch("/me", summary="Update time zone, name and preferences")
def update_me(body: PrefsIn, c: Caller = Depends(caller)):
    upd = {}
    if body.timezone:
        upd["timezone"] = body.timezone if core.safe_tz(body.timezone).key == body.timezone else "UTC"
    for k in ("displayName", "wakeTime"):
        if getattr(body, k) is not None:
            upd[k] = getattr(body, k)
    prefs = {k: getattr(body, k) for k in ("blend", "lucilleWhisper", "defaultBrainwave") if getattr(body, k) is not None}
    if prefs:
        upd["prefs"] = prefs
    if body.birthYear:
        age = core.now_utc().year - body.birthYear
        upd["ageBand"] = "under13" if age < 13 else ("13-17" if age < 18 else "18+")
        upd["isMinor"] = age < 18
    c.repo.set(user_path(c.uid, "state"), upd, merge=True)
    return me(c)


@router.get("/me/progress", summary="Unified streak, coins and Energy Scan unlocks (1,000 coins per energy center)")
def progress(c: Caller = Depends(caller)):
    st = core.get_state(c.repo, c.uid)
    return {"streakDays": st["streakDays"], "bestStreak": st["bestStreak"], "recentDays": st.get("recentDays", []),
            "coins": st["coins"], "lifetimeCoins": st["lifetimeCoins"],
            "energy": core.energy_progress(int(st.get("lifetimeCoins", 0))),
            "coinRules": {k: {"coins": v[0], "dailyCap": v[1]} for k, v in core.COIN_RULES.items()}}


@router.get("/energy/centers", summary="The 7 energy centers (Grounding -> Purpose) and unlock thresholds")
def centers(c: Caller = Depends(caller)):
    st = core.get_state(c.repo, c.uid)
    return core.energy_progress(int(st.get("lifetimeCoins", 0)))


@router.get("/home", summary="Home in one call: Mood Orb, Self-Care Score, streak, coins, today's plan, a suggestion")
def home(c: Caller = Depends(caller), period: Literal["week", "month", "year"] = "week",
         lat: Optional[float] = None, lon: Optional[float] = None):
    st = core.get_state(c.repo, c.uid)
    night = core.is_night(c.tz, lat, lon)
    word = (st.get("lastMood") or {}).get("word", "Neutral")
    plan = selfcare.active_plan(c.repo, c.uid)
    return {
        "displayName": st.get("displayName"),
        "phase": core.day_phase(c.tz), "night": night,
        "lastMood": st.get("lastMood"),
        "orb": mood.summary(c.repo, c.uid, c.tz, period, night),
        "score": selfcare.compute_score(c.repo, c.uid, c.tz),
        "streakDays": st["streakDays"], "coins": st["coins"],
        "energy": core.energy_progress(int(st.get("lifetimeCoins", 0))),
        "todayPlan": selfcare.today_items(plan, c.tz),
        "suggestion": (mood.suggestions_for(word, c.tz, 1) or [None])[0],
        "aiDisclosure": core.AI_DISCLOSURE,
    }


@router.get("/suggestions", summary="Lucille Suggestions for any screen. Every link is allowlisted server-side")
def suggestions(c: Caller = Depends(caller),
                context: Literal["home", "mood_result", "journal_saved", "session_complete", "explore"] = "home",
                word: Optional[str] = None, limit: int = Query(3, ge=1, le=6)):
    st = core.get_state(c.repo, c.uid)
    word = word or (st.get("lastMood") or {}).get("word", "Neutral")
    exclude = {"journal_saved": {"journal"}, "session_complete": {"listen"}}.get(context)
    return {"context": context, "word": word, "items": mood.suggestions_for(word, c.tz, limit, exclude)}
