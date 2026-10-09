"""
/v1/soundscapes — Soundscapes AI: the 9 saved music categories, modes (Lucille Picks, Focus, Calm,
Sleep, Move), Mood Field, Play-for-right-now, Inner Weather, Daily Drop, Compose, sessions, library.

The legacy /soundscapes/* endpoints in main.py stay as they are (the current app calls them).
"""

from __future__ import annotations

import hmac
import logging
from datetime import timedelta
from typing import List, Literal, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field

from escape_api import core, mood
from escape_api import soundscapes as ss
from escape_api.deps import Caller, caller, has_consent
from escape_api.repo import get_repo, user_path
from escape_api.settings import cfg

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/soundscapes", tags=["v1 · soundscapes"])

Mode = Literal["picks", "focus", "calm", "sleep", "move"]
CategoryId = Literal["music_meditations", "vaporwave", "jazz", "nature", "binaural_beats", "brainwave_music",
                     "raw_frequencies", "sleep_ambient", "depression_anxiety"]


class MoodField(BaseModel):
    energy: float = Field(0.5, ge=0, le=1, description="calm 0 -> charged 1")
    texture: float = Field(0.5, ge=0, le=1, description="grounded 0 -> dreamy 1")


class PickIn(BaseModel):
    mode: Mode = "picks"
    moodField: Optional[MoodField] = None
    lat: Optional[float] = Field(None, ge=-90, le=90)
    lon: Optional[float] = Field(None, ge=-180, le=180)


class ComposeIn(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=400)
    mode: Mode = "calm"
    minutes: int = Field(30, ge=5, le=240)
    brainwave: Optional[Literal["delta", "theta", "alpha", "beta"]] = None
    moodField: MoodField = Field(default_factory=MoodField)
    useInnerWeather: bool = False
    lat: Optional[float] = None
    lon: Optional[float] = None


class SessionIn(BaseModel):
    trackId: Optional[str] = None
    compositionId: Optional[str] = None
    mode: Mode = "calm"
    startedFrom: Literal["playNow", "mode", "category", "compose", "dailyDrop", "journal", "suggestion", "library"] = "mode"
    timerMinutes: Optional[int] = Field(None, ge=5, le=240)
    moodField: Optional[MoodField] = None
    moodBefore: Optional[str] = Field(None, description="Mood word from the 10-second check-in")


class SessionDoneIn(BaseModel):
    minutes: float = Field(..., ge=0, le=600)
    completed: bool = True
    moodAfter: Optional[str] = None


class RenderCallbackIn(BaseModel):
    status: Literal["rendering", "ready", "failed"]
    segments: List[dict] = Field(default_factory=list)
    measured: Optional[dict] = None
    failReason: Optional[str] = None


def _premium(c: Caller) -> bool:
    return bool(core.get_state(c.repo, c.uid).get("isPremium"))


# ───────────────────────── catalog ─────────────────────────

@router.get("/categories", summary="The 9 music categories (static; cache 24h)")
def categories():
    return {"categories": ss.categories(), "modes": ss.catalog()["modes"]}


@router.get("/catalog", summary="Browse tracks by category / mode / search. Audio = intro + 4 bodies + outro")
def browse(c: Caller = Depends(caller), category: Optional[CategoryId] = None, mode: Optional[Mode] = None,
           q: Optional[str] = Query(None, max_length=60), word: Optional[str] = None,
           limit: int = Query(40, ge=1, le=100)):
    night = core.is_night(c.tz)
    items = [ss.track_out(t, night) for t in ss.search(category, None if mode == "picks" else mode, q, word)[:limit]]
    return {"items": items, "count": len(items)}


@router.get("/tracks/{track_id}", summary="One track with segment URLs and its Mood Orb loop")
def track(track_id: str, c: Caller = Depends(caller)):
    t = ss.find_track(track_id)
    if not t:
        raise HTTPException(404, "Track not found")
    out = ss.track_out(t, core.is_night(c.tz))
    out["locked"] = t["premium"] and not _premium(c)
    cat = next(x for x in ss.catalog()["categories"] if x["id"] == t["category"])
    out["disclaimer"] = cat.get("disclaimer")
    return out


@router.get("/visual", summary="Nearest of the 36 Mood Orb loops for a mode + Mood Field (+ night twin)")
def visual(c: Caller = Depends(caller), mode: Mode = "picks", energy: float = Query(0.5, ge=0, le=1),
           texture: float = Query(0.5, ge=0, le=1), lat: Optional[float] = None, lon: Optional[float] = None):
    return core.soundscape_loop(mode, energy, texture, core.is_night(c.tz, lat, lon))


# ───────────────────────── home, inner weather, picks, daily drop ─────────────────────────

@router.get("/inputs-now", summary="Inner Weather: day phase, sun, night flag, weather (cache 15 min)")
def inputs_now(c: Caller = Depends(caller), lat: Optional[float] = Query(None, ge=-90, le=90),
               lon: Optional[float] = Query(None, ge=-180, le=180)):
    st = core.get_state(c.repo, c.uid)
    out = {"phase": core.day_phase(c.tz), "night": core.is_night(c.tz, lat, lon), "timezone": c.tz,
           "mood": st.get("lastMood"), "weather": None, "sun": None}
    if lat is not None and lon is not None and has_consent(c, "location_weather"):
        from datetime import date
        rise, sset = core.sun_times(lat, lon, date.fromisoformat(core.local_day(c.tz)))
        out["sun"] = {"sunrise": core.iso(rise) if rise else None, "sunset": core.iso(sset) if sset else None}
        out["weather"] = ss.weather(lat, lon)
    out["inputsUsed"] = [k for k in ("phase", "mood", "weather") if out.get(k)]
    return out


def _do_pick(c: Caller, body: PickIn) -> dict:
    st = core.get_state(c.repo, c.uid)
    word = (st.get("lastMood") or {}).get("word")
    night = core.is_night(c.tz, body.lat, body.lon)
    mode, why = (body.mode, None) if body.mode != "picks" else ss.auto_mode(c.tz, word, night)
    mf = body.moodField or MoodField(
        energy=(st.get("lastMood") or {}).get("energy", 0.5) if mode != "sleep" else 0.15,
        texture=0.6 if mode in ("sleep", "calm") else 0.35)
    t = ss.pick(mode, mf.energy, mf.texture, word, _premium(c))
    return {"mode": mode, "why": why or f"Matched to your Mood Field for {mode}.", "moodField": mf.model_dump(),
            "track": ss.track_out(t, night), "aiLabel": "Picked by Lucille (AI)"}


@router.post("/pick", summary="Play for right now (Lucille Picks) or best match for a mode + Mood Field")
def pick(body: PickIn, c: Caller = Depends(caller)):
    return _do_pick(c, body)


def _daily_drop(c: Caller, lat: Optional[float], lon: Optional[float]) -> dict:
    """Per cohort (date x tz x weather bucket), never per user — a couple dozen drops a day, not one per user."""
    day = core.local_day(c.tz)
    bucket = ss.weather(lat, lon)["bucket"] if (lat is not None and lon is not None and has_consent(c, "location_weather")) else "any"
    did = f"{day}_{c.tz.replace('/', '-')}_{bucket}"
    repo = get_repo()
    doc = repo.get(f"escape_daily_drops/{did}")
    if not doc:
        import random
        rnd = random.Random(did)
        mode = "calm" if bucket in ("rain", "snow", "cloudy") else "focus"
        t = rnd.choice([t for t in ss.catalog()["tracks"] if t["mode"] == mode and not t["premium"]])
        title = {"rain": "Drizzle", "snow": "Snowlight", "cloudy": "Grey Morning", "clear": "Clear Sky",
                 "heat": "Heat Haze"}.get(bucket, "Today")
        doc = {"id": did, "day": day, "bucket": bucket, "title": f"{core.parse_iso(core.iso(core.now_utc())).astimezone(core.safe_tz(c.tz)).strftime('%A')} {title}",
               "trackId": t["id"], "source": "catalog", "createdAt": core.iso(core.now_utc())}
        repo.set(f"escape_daily_drops/{did}", doc)
    t = ss.find_track(doc["trackId"]) or ss.catalog()["tracks"][0]
    return {**doc, "track": ss.track_out(t, core.is_night(c.tz, lat, lon))}


@router.get("/daily-drop", summary="Lucille's Daily Drop for your cohort (date x time zone x weather)")
def daily_drop(c: Caller = Depends(caller), lat: Optional[float] = None, lon: Optional[float] = None):
    return _daily_drop(c, lat, lon)


@router.get("/home", summary="Soundscapes home in one call: play-now pick, inner weather, daily drop, categories, recents")
def home(c: Caller = Depends(caller), lat: Optional[float] = None, lon: Optional[float] = None):
    night = core.is_night(c.tz, lat, lon)
    st = core.get_state(c.repo, c.uid)
    recents = [d for _, d in c.repo.query(user_path(c.uid, "listening_sessions"), order_by="startedAt", desc=True, limit=3)]
    rows = []
    for cat in ss.categories():
        tracks = [ss.track_out(t, night) for t in ss.search(cat["id"], word=(st.get("lastMood") or {}).get("word"))[:4]]
        rows.append({"category": cat, "tracks": tracks})
    return {
        "playNow": _do_pick(c, PickIn(mode="picks", lat=lat, lon=lon)),
        "inner": inputs_now(c, lat, lon),
        "dailyDrop": _daily_drop(c, lat, lon),
        "modes": ss.catalog()["modes"],
        "rows": rows,
        "recents": recents,
        "streakDays": st["streakDays"], "night": night,
        "checkInDue": st.get("lastCheckInDay") != core.local_day(c.tz) and not night,
    }


# ───────────────────────── compose (Lucille renders) ─────────────────────────

async def _send_render(uid: str, comp: dict) -> None:
    import httpx
    s = cfg()
    payload = {
        "compositionId": comp["id"], "mode": comp["mode"], "prompt": comp["prompt"], "recipe": comp["recipe"],
        "segments": {"introSec": 45, "bodyCount": 4, "bodySec": 150, "outroSec": 60, "sameKeyAndTempo": True},
        "output": {"codec": "aac-lc", "container": "m4a", "sampleRate": 48000, "channels": 2, "bitrateKbps": 160,
                   "loudnessLufs": comp["recipe"]["loudnessLufs"], "truePeakDbtp": -1},
        "callbackUrl": f"{s.public_base_url.rstrip('/')}/v1/soundscapes/render-callback/{uid}/{comp['id']}",
        "energy": comp["moodField"]["energy"],
    }
    try:
        async with httpx.AsyncClient(timeout=15) as cl:
            r = await cl.post(s.render_url, json=payload, headers={"X-Api-Key": s.render_api_key})
            r.raise_for_status()
        get_repo().set(user_path(uid, "compositions", comp["id"]), {"status": "rendering"}, merge=True)
    except Exception as e:
        logger.warning(f"render request failed: {e}")
        get_repo().set(user_path(uid, "compositions", comp["id"]),
                       {"status": "failed", "failReason": "render_unavailable"}, merge=True)


@router.post("/compose", status_code=202, summary="Lucille Compose: queue a personal soundscape render (daily limit)")
async def compose(body: ComposeIn, bg: BackgroundTasks, c: Caller = Depends(caller)):
    limit = cfg().quota_compose_per_day_premium if _premium(c) else cfg().quota_compose_per_day_free
    if not core.check_quota(c.repo, c.uid, c.tz, "compose", limit):
        raise HTTPException(429, {"code": "daily_limit", "limit": limit, "upgrade": not _premium(c)})
    safety = core.safety_check(body.prompt)
    w = ss.weather(body.lat, body.lon) if (body.useInnerWeather and body.lat is not None and body.lon is not None
                                           and has_consent(c, "location_weather")) else {"bucket": None}
    phase = core.day_phase(c.tz)
    mode = body.mode if body.mode != "picks" else ss.auto_mode(c.tz, None, core.is_night(c.tz))[0]
    rec = ss.recipe(mode, body.moodField.energy, body.moodField.texture, w.get("bucket"), phase, body.brainwave)
    title = " ".join(body.prompt.split()[:4]).strip(".,!").title()
    comp = {
        "id": core.new_id("cmp"), "source": "compose", "prompt": body.prompt, "mode": mode, "minutes": body.minutes,
        "brainwave": rec["brainwave"], "moodField": body.moodField.model_dump(), "recipe": rec,
        "status": "queued", "title": title,
        "whyThisSound": f"{rec['tempoBpm']} bpm, {rec['layers']['ambience'].replace('_', ' ')} bed"
                        + (f", a {rec['brainwave']['hz']} Hz {rec['brainwave']['type']} layer" if rec["brainwave"] else "")
                        + f", tuned for {phase.lower()}.",
        "visual": core.soundscape_loop(mode, body.moodField.energy, body.moodField.texture, core.is_night(c.tz)),
        "createdAt": core.iso(core.now_utc()), "aiLabel": "Composed by Lucille (AI)",
        "flagged": safety["crisis"] or None,
    }
    c.repo.set(user_path(c.uid, "compositions", comp["id"]), comp)
    if cfg().render_url:
        bg.add_task(_send_render, c.uid, comp)
    return {"compositionId": comp["id"], "status": comp["status"], "etaSec": 120, "composition": comp,
            "renderConfigured": bool(cfg().render_url), "safety": safety if safety["crisis"] else None}


@router.get("/compositions", summary="Your compositions, newest first")
def compositions(c: Caller = Depends(caller), limit: int = Query(20, ge=1, le=50)):
    return {"items": [d for _, d in c.repo.query(user_path(c.uid, "compositions"), order_by="createdAt", desc=True, limit=limit)]}


@router.get("/compositions/{cid}", summary="Poll a composition (or listen to the Firestore doc in real time)")
def composition(cid: str, c: Caller = Depends(caller)):
    d = c.repo.get(user_path(c.uid, "compositions", cid))
    if not d:
        raise HTTPException(404, "Not found")
    return d


@router.post("/render-callback/{uid}/{cid}", include_in_schema=False)
def render_callback(uid: str, cid: str, body: RenderCallbackIn,
                    x_lucille_secret: Optional[str] = Header(default=None, alias="X-Lucille-Secret")):
    secret = cfg().render_callback_secret
    if not secret or not x_lucille_secret or not hmac.compare_digest(secret, x_lucille_secret):
        raise HTTPException(401, "bad secret")
    repo = get_repo()
    if not repo.get(user_path(uid, "compositions", cid)):
        raise HTTPException(404, "unknown composition")
    repo.set(user_path(uid, "compositions", cid),
             {"status": body.status, "segments": body.segments, "measured": body.measured,
              "failReason": body.failReason, "readyAt": core.iso(core.now_utc())}, merge=True)
    return {"ok": True}


# ───────────────────────── sessions ─────────────────────────

@router.post("/sessions", status_code=201, summary="Start a listening session (optionally with the 10-second mood check-in)")
def start_session(body: SessionIn, c: Caller = Depends(caller)):
    if not body.trackId and not body.compositionId:
        raise HTTPException(422, "trackId or compositionId required")
    if body.trackId and not ss.find_track(body.trackId):
        raise HTTPException(404, "Track not found")
    sid = core.new_id("ls")
    doc = {"id": sid, **body.model_dump(exclude_none=True), "startedAt": core.iso(core.now_utc()), "completed": False}
    if body.moodBefore and has_consent(c, "wellbeing_data"):
        r = mood.fuse_inputs(word=body.moodBefore)
        ck = mood.record_checkin(c.repo, c.uid, c.tz, r, "session_before", extra={"sessionId": sid})
        doc["moodBefore"] = {"word": ck["word"], "checkinId": ck["id"]}
        c.repo.set(user_path(c.uid, "state"), {"lastCheckInDay": core.local_day(c.tz)}, merge=True)
    c.repo.set(user_path(c.uid, "listening_sessions", sid), doc)
    return {"session": doc}


@router.post("/sessions/{sid}/complete", summary="End a session: streak, coins (1 per 5 min), before/after mood")
def complete_session(sid: str, body: SessionDoneIn, c: Caller = Depends(caller)):
    s = c.repo.get(user_path(c.uid, "listening_sessions", sid))
    if not s:
        raise HTTPException(404, "Session not found")
    if s.get("completedAt"):
        return {"session": s, "alreadyCompleted": True}
    upd = {"minutes": round(body.minutes, 1), "completed": body.completed, "completedAt": core.iso(core.now_utc())}
    after = None
    if body.moodAfter and has_consent(c, "wellbeing_data"):
        r = mood.fuse_inputs(word=body.moodAfter)
        ck = mood.record_checkin(c.repo, c.uid, c.tz, r, "session_after", extra={"sessionId": sid})
        after = {"word": ck["word"], "checkinId": ck["id"]}
        upd["moodAfter"] = after
    coins, streak = 0, core.get_state(c.repo, c.uid)["streakDays"]
    if body.minutes >= 5:
        streak = core.touch_activity(c.repo, c.uid, c.tz, "soundscape")["streakDays"]
        coins = core.award_coins(c.repo, c.uid, c.tz, "listening_5min", int(body.minutes // 5))
    c.repo.set(user_path(c.uid, "listening_sessions", sid), upd, merge=True)
    return {"session": {**s, **upd}, "moodBefore": s.get("moodBefore"), "moodAfter": after,
            "streakDays": streak, "coinsAwarded": coins,
            "showFeedbackCard": s.get("mode") != "sleep"}


@router.get("/sessions", summary="Listening history")
def sessions(c: Caller = Depends(caller), limit: int = Query(20, ge=1, le=100), cursor: Optional[str] = None):
    items = [d for _, d in c.repo.query(user_path(c.uid, "listening_sessions"), order_by="startedAt", desc=True,
                                        limit=limit, start_after=cursor)]
    return {"items": items, "nextCursor": items[-1]["startedAt"] if len(items) == limit else None}


# ───────────────────────── library ─────────────────────────

@router.get("/library", summary="Saved tracks and compositions")
def library(c: Caller = Depends(caller)):
    night = core.is_night(c.tz)
    out = []
    for _, d in c.repo.query(user_path(c.uid, "library"), order_by="savedAt", desc=True):
        t = ss.find_track(d["id"])
        out.append({**d, "track": ss.track_out(t, night) if t else None})
    return {"items": out}


@router.put("/library/{item_id}", summary="Save a track (trackId) or composition (cmp_...) to the Library")
def save(item_id: str, c: Caller = Depends(caller)):
    kind = "composition" if item_id.startswith("cmp_") else "track"
    if kind == "track" and not ss.find_track(item_id):
        raise HTTPException(404, "Track not found")
    doc = {"id": item_id, "kind": kind, "savedAt": core.iso(core.now_utc())}
    c.repo.set(user_path(c.uid, "library", item_id), doc)
    return doc


@router.delete("/library/{item_id}", status_code=204)
def unsave(item_id: str, c: Caller = Depends(caller)):
    c.repo.delete(user_path(c.uid, "library", item_id))
