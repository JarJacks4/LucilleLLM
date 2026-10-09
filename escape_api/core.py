"""
Shared logic for the v1 API: datasets, time zones, mood math, streak + coins,
quotas, deep-link allowlist, asset URLs, safety. No FastAPI here so it is easy
to unit-test.
"""

from __future__ import annotations

import json
import math
import re
import os
import uuid
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from escape_api.repo import Repo, user_path
from escape_api.settings import cfg

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


# ───────────────────────── datasets ─────────────────────────

@lru_cache(maxsize=None)
def dataset(name: str) -> Dict[str, Any]:
    with open(os.path.join(DATA_DIR, f"{name}.json"), encoding="utf-8") as f:
        return json.load(f)


def mood_words() -> List[Dict[str, Any]]:
    return dataset("mood_words")["words"]


@lru_cache(maxsize=None)
def _word_index() -> Dict[str, Dict[str, Any]]:
    return {w["word"].lower(): w for w in mood_words()}


def find_word(word: Optional[str]) -> Optional[Dict[str, Any]]:
    return _word_index().get((word or "").strip().lower()) if word else None


# ───────────────────────── time ─────────────────────────

def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


_OFFSET = re.compile(r"^(?:UTC|GMT)?\s*([+-])(\d{1,2})(?::?(\d{2}))?$")


def safe_tz(tz: Optional[str]):
    """IANA name ('America/New_York') or a UTC offset ('+05:30', 'UTC-04:00'). Falls back to UTC."""
    if not tz:
        return ZoneInfo("UTC")
    m = _OFFSET.match(tz.strip())
    if m:
        mins = int(m.group(2)) * 60 + int(m.group(3) or 0)
        if mins <= 14 * 60:
            return timezone(timedelta(minutes=mins if m.group(1) == "+" else -mins))
    try:
        return ZoneInfo(tz)
    except Exception:
        return ZoneInfo("UTC")


def local_day(tz: Optional[str], at: Optional[datetime] = None) -> str:
    return (at or now_utc()).astimezone(safe_tz(tz)).date().isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ───────────────────────── sun + night (Soundscapes rule) ─────────────────────────

def sun_times(lat: float, lon: float, day: date) -> Tuple[Optional[datetime], Optional[datetime]]:
    """NOAA approximation, UTC sunrise/sunset. None when polar day/night."""
    n = day.timetuple().tm_yday
    g = 2 * math.pi / 365 * (n - 1)
    eqt = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                    - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g)
            + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    lat_r = math.radians(lat)
    cos_ha = (math.cos(math.radians(90.833)) / (math.cos(lat_r) * math.cos(decl))
              - math.tan(lat_r) * math.tan(decl))
    if cos_ha < -1 or cos_ha > 1:
        return None, None
    ha = math.degrees(math.acos(cos_ha))
    base = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    rise = base + timedelta(minutes=720 - 4 * (lon + ha) - eqt)
    sset = base + timedelta(minutes=720 - 4 * (lon - ha) - eqt)
    return rise, sset


def is_night(tz: Optional[str], lat: Optional[float] = None, lon: Optional[float] = None,
             at: Optional[datetime] = None) -> bool:
    """Local sunset -> sunrise; fallback 8pm-6am (same rule as Soundscapes backgrounds)."""
    at = at or now_utc()
    if lat is not None and lon is not None:
        local_date = at.astimezone(safe_tz(tz)).date()
        rise, sset = sun_times(lat, lon, local_date)
        if rise and sset:
            return not (rise <= at <= sset)
    h = at.astimezone(safe_tz(tz)).hour
    return h >= 20 or h < 6


DAY_PHASES = [
    (5, "Morning Rise"), (10, "Late-Morning Clarity"), (13, "Afternoon Lift"),
    (17, "Golden Hour Unwind"), (21, "Night Drift"), (24, "Deep Night"),
]


def day_phase(tz: Optional[str], at: Optional[datetime] = None) -> str:
    h = (at or now_utc()).astimezone(safe_tz(tz)).hour
    if h < 5:
        return "Deep Night"
    for end, name in DAY_PHASES:
        if h < end:
            return name
    return "Deep Night"


# ───────────────────────── mood math ─────────────────────────

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def tone_for(valence: float, energy: float, grateful: bool = False) -> str:
    """Same mapping as the Journal visuals manifest.json."""
    if valence >= 0.55:
        return "tender" if grateful and energy < 0.55 else ("bright" if energy >= 0.55 else "calm")
    if valence <= 0.45:
        return "restless" if energy >= 0.55 else "heavy"
    return "tender" if grateful else "mixed"


def nearest_word(valence: float, energy: float) -> Dict[str, Any]:
    return min(mood_words(), key=lambda w: (w["valence"] - valence) ** 2 + (w["energy"] - energy) ** 2)


@lru_cache(maxsize=None)
def _derived_keywords() -> Dict[str, List[str]]:
    try:
        return dataset("mood_keywords_derived")["words"]
    except Exception:
        return {}


def interpret_text(text: str) -> Optional[Dict[str, Any]]:
    """Free (no LLM) keyword read of a short mood description. Returns best word or None."""
    t = f" {text.lower()} "
    best, score = None, 0
    for w in mood_words():
        hits = sum(1 for k in w["keywords"] if f" {k}" in t or f"{k} " in t)
        if hits > score:
            best, score = w, hits
    if not best:
        # second pass: dataset-derived keywords (scripts/datasets/build_mood_keywords.py), only
        # when the hand-written list found nothing, since they are noisier
        derived = _derived_keywords()
        tally = {w: sum(1 for k in ks if f" {k} " in t) for w, ks in derived.items()}
        top = max(tally.items(), key=lambda kv: kv[1], default=(None, 0))
        if top[1] >= 1:
            best = find_word(top[0])
    if not best:
        return None
    # negation flips toward the opposite side of valence
    if any(n in t for n in (" not ", " no longer ", " isn't ", " don't feel ")) and best["valence"] >= 0.55:
        return nearest_word(1 - best["valence"], best["energy"])
    return best


def family_meta(family: str) -> Dict[str, Any]:
    return dataset("mood_words")["families"].get(family, dataset("mood_words")["families"]["mixed"])


def orb_assets(tone: str, night: bool = False) -> Dict[str, str]:
    base = cfg().asset_base_url.rstrip("/") + "/journal"
    stem = f"{base}/journal_orb_{tone}"
    wash = {"bright": "calmwash", "calm": "calmwash", "tender": "calmwash",
            "restless": "warmwash", "heavy": "coolwash", "mixed": "coolwash"}[tone]
    bg = f"{base}/journal_bg_{wash}{'_night' if night else ''}"
    return {
        "tone": tone,
        "accent": dataset("mood_words")["tones"][tone]["accent"],
        "orbHevc": f"{stem}_hevc.mp4", "orbH264": f"{stem}_h264.mp4",
        "orbPoster": f"{stem}_poster.webp", "orbCard": f"{stem}_card.webp",
        "background": wash + ("_night" if night else ""),
        "bgHevc": f"{bg}_hevc.mp4", "bgH264": f"{bg}_h264.mp4",
        "bgPoster": f"{bg}_poster.webp", "bgOffline": f"{bg}_540_offline.mp4",
        "night": night,
    }


def soundscape_loop(mode: str, energy: float, texture: float, night: bool = False, version: int = 3) -> Dict[str, str]:
    """Nearest of the 36 Mood Orb loops: escape_{mode}_e{XX}_t{XX}_v{N}."""
    m = mode if mode in ("focus", "calm", "sleep", "picks") else "picks"
    snap = lambda v: min((15, 50, 85), key=lambda c: abs(c / 100 - v))  # noqa: E731
    name = f"escape_{m}_e{snap(energy):02d}_t{snap(texture):02d}_v{version}{'_night' if night else ''}"
    base = cfg().asset_base_url.rstrip("/") + "/soundscapes/loops"
    return {"name": name, "hevc": f"{base}/{name}_hevc.mp4", "h264": f"{base}/{name}_h264.mp4",
            "poster": f"{base}/{name}_poster.webp", "card": f"{base}/{name}_card.webp",
            "offline": f"{base}/{name}_540_offline.mp4"}


# ───────────────────────── user state: streak, coins, energy ─────────────────────────

ENERGY_CENTERS = [
    {"level": 1, "key": "grounding",  "name": "Grounding",  "chakra": "Root",         "color": "#E5484D", "route": "MindRootChakraVersion5"},
    {"level": 2, "key": "creativity", "name": "Creativity", "chakra": "Sacral",       "color": "#EF7702", "route": "MindSacralChakraVersion5"},
    {"level": 3, "key": "power",      "name": "Power",      "chakra": "Solar Plexus", "color": "#F5C542", "route": "MindSolarPlexusChakraVersion5"},
    {"level": 4, "key": "connection", "name": "Connection", "chakra": "Heart",        "color": "#39D9C1", "route": "MindHeartChakraVersion5"},
    {"level": 5, "key": "expression", "name": "Expression", "chakra": "Throat",       "color": "#4CF6F6", "route": "MindThroatChakraVersion5"},
    {"level": 6, "key": "intuition",  "name": "Intuition",  "chakra": "Third Eye",    "color": "#8E7CD9", "route": "MindThirdEyeChakraVersion5"},
    {"level": 7, "key": "purpose",    "name": "Purpose",    "chakra": "Crown",        "color": "#B9A3F0", "route": "MindCrownChakraVersion5"},
]

# coins per action, daily cap per action (anti-farming)
COIN_RULES: Dict[str, Tuple[int, int]] = {
    "mood_checkin": (5, 3),
    "journal_entry": (15, 3),
    "journal_reflect": (5, 3),
    "gratitude": (10, 1),
    "letter": (10, 1),
    "listening_5min": (1, 24),
    "plan_item": (10, 6),
    "energy_scan": (10, 1),
}


def energy_progress(lifetime_coins: int) -> Dict[str, Any]:
    step = cfg().energy_level_step
    level = min(len(ENERGY_CENTERS), lifetime_coins // step)
    current = ENERGY_CENTERS[level - 1] if level >= 1 else None
    nxt = ENERGY_CENTERS[level] if level < len(ENERGY_CENTERS) else None
    return {
        "energyScanUnlocked": level >= 1,
        "level": level,
        "current": current,
        "next": nxt,
        "coinsToNext": (nxt["level"] * step - lifetime_coins) if nxt else 0,
        "step": step,
        "centers": [dict(c, unlocked=c["level"] <= level, unlocksAt=c["level"] * step) for c in ENERGY_CENTERS],
    }


def get_state(repo: Repo, uid: str) -> Dict[str, Any]:
    st = repo.get(user_path(uid, "state")) or {}
    st.setdefault("streakDays", 0)
    st.setdefault("bestStreak", 0)
    st.setdefault("coins", 0)
    st.setdefault("lifetimeCoins", 0)
    st.setdefault("timezone", "UTC")
    return st


def touch_activity(repo: Repo, uid: str, tz: Optional[str], kind: str) -> Dict[str, Any]:
    """
    One streak across mind / body / journal / sound (roadmap: 'one streak number').
    Same local day never double-counts. One grace day per 7 days keeps a missed
    day from wiping the streak ('gentle rhythm, not streaks').
    """
    st = get_state(repo, uid)
    tz = tz or st.get("timezone") or "UTC"
    today = date.fromisoformat(local_day(tz))
    last = date.fromisoformat(st["lastActiveDay"]) if st.get("lastActiveDay") else None
    streak = st["streakDays"]
    grace_used = st.get("graceUsedOn")
    if last == today:
        pass
    elif last == today - timedelta(days=1):
        streak += 1
    elif last == today - timedelta(days=2) and (not grace_used or date.fromisoformat(grace_used) <= today - timedelta(days=7)):
        streak += 1
        grace_used = today.isoformat()
    else:
        streak = 1
    active_days = [d for d in st.get("recentDays", []) if d >= (today - timedelta(days=27)).isoformat()]
    if today.isoformat() not in active_days:
        active_days.append(today.isoformat())
    upd = {
        "streakDays": streak, "bestStreak": max(streak, st["bestStreak"]),
        "lastActiveDay": today.isoformat(), "graceUsedOn": grace_used, "timezone": tz,
        "recentDays": sorted(active_days)[-28:], "updatedAt": iso(now_utc()),
        f"lastActivity.{kind}": iso(now_utc()),
    }
    flat = {k: v for k, v in upd.items() if "." not in k}
    flat["lastActivity"] = dict(st.get("lastActivity", {}), **{kind: upd[f"lastActivity.{kind}"]})
    repo.set(user_path(uid, "state"), flat, merge=True)
    st.update(flat)
    return st


def award_coins(repo: Repo, uid: str, tz: Optional[str], action: str, units: int = 1) -> int:
    """Server-side only. Respects per-day caps. Returns coins actually awarded."""
    if action not in COIN_RULES or units <= 0:
        return 0
    per, cap = COIN_RULES[action]
    day = local_day(tz)
    usage_path = user_path(uid, "usage", day)
    used = int((repo.get(usage_path) or {}).get(f"coins_{action}", 0))
    allowed = max(0, min(units, cap - used))
    if not allowed:
        return 0
    amount = per * allowed
    repo.increment(usage_path, {f"coins_{action}": allowed})
    repo.increment(user_path(uid, "state"), {"coins": amount, "lifetimeCoins": amount})
    return amount


def check_quota(repo: Repo, uid: str, tz: Optional[str], kind: str, limit: int) -> bool:
    """Increment-and-check a per-day counter. False = over quota (no money spent)."""
    path = user_path(uid, "usage", local_day(tz))
    used = int((repo.get(path) or {}).get(kind, 0))
    if used >= limit:
        return False
    repo.increment(path, {kind: 1})
    return True


# ───────────────────────── deep links ─────────────────────────

def resolve_route(name: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Allowlist check. Planned pages fall back to their live fallback unless allowed."""
    routes = dataset("deeplinks")["routes"]
    r = routes.get(name)
    if not r:
        return None
    if r["status"] != "live" and not cfg().deeplinks_allow_planned:
        fb = r.get("fallback")
        return resolve_route(fb, None) if fb else None
    from urllib.parse import urlencode
    q = f"?{urlencode(params)}" if params else ""
    return {"route": name, "path": r["path"], "params": params or {},
            "url": f"{dataset('deeplinks')['scheme']}{r['path'].lstrip('/')}{q}"}


# ───────────────────────── safety ─────────────────────────

def safety_check(text: str) -> Dict[str, Any]:
    """Reuse the existing keyword safety screen (no API call)."""
    if not text or not text.strip():
        return {"crisis": False, "riskLevel": "low", "resources": []}
    try:
        from safety_service import get_safety_service
        svc = get_safety_service()
        res = svc.check_input(text)
        crisis = bool(res.crisis_detected or res.risk_level.value in ("high", "critical"))
        resources = [r.model_dump() if hasattr(r, "model_dump") else dict(r) for r in svc.get_crisis_resources()] if crisis else []
        return {"crisis": crisis, "riskLevel": res.risk_level.value, "flags": res.flags, "resources": resources}
    except Exception:
        low = text.lower()
        crisis = any(k in low for k in ("kill myself", "suicide", "end my life", "self harm", "hurt myself"))
        return {"crisis": crisis, "riskLevel": "critical" if crisis else "low",
                "resources": [{"name": "988 Suicide & Crisis Lifeline", "phone": "988", "text": "988",
                               "url": "https://988lifeline.org"}] if crisis else []}


AI_DISCLOSURE = ("Lucille is an AI companion, not a person or a licensed therapist. "
                 "She can't diagnose or treat anything. If you're in crisis, call or text 988 (US) or your local emergency number.")
