"""
Self-Care Score (0-100) and Self-Care Plans.

Score: five components from data we already roll up, so it costs a handful of reads
and no LLM. Missing components are dropped and their weight redistributed.

Plans: built ONLY from the activity library (escape_api/data/selfcare_activities.json),
so every item is a real in-app deep link. PlanProvider is the seam for later:
'catalog' (today, free, deterministic) and 'claude' (Anthropic Messages API picks and
orders activity ids from the same library; output is validated against it).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from escape_api import core
from escape_api.repo import Repo, user_path
from escape_api.settings import cfg

logger = logging.getLogger(__name__)

WEIGHTS = {"mood": 0.25, "rhythm": 0.2, "reflection": 0.2, "practice": 0.2, "plan": 0.15}
BANDS = [(80, "Thriving"), (60, "Steady"), (40, "Finding rhythm"), (0, "Rebuilding")]
LEVER = {"mood": "mood_scan", "rhythm": "journal_ritual", "reflection": "journal_free",
         "practice": "soundscape_pick", "plan": "breath_basic"}


def _since(tz: str, days: int) -> str:
    return (date.fromisoformat(core.local_day(tz)) - timedelta(days=days - 1)).isoformat()


def compute_score(repo: Repo, uid: str, tz: str) -> Dict[str, Any]:
    today = core.local_day(tz)
    cached = repo.get(user_path(uid, "scores", today))
    st = core.get_state(repo, uid)
    if cached and cached.get("rev") == _rev(st):
        return cached
    s14 = _since(tz, 14)
    days = [d for _, d in repo.query(user_path(uid, "mood_days"), where=[("day", ">=", s14)])]
    comp: Dict[str, Optional[float]] = {}
    detail: Dict[str, str] = {}

    n = sum(d.get("count", 0) for d in days)
    if n:
        avg_v = sum(d.get("sumV", 0) for d in days) / n
        daily = [d["sumV"] / d["count"] for d in days if d.get("count")]
        spread = (max(daily) - min(daily)) if len(daily) > 1 else 0
        comp["mood"] = max(0.0, min(1.0, avg_v * 0.85 + (1 - spread) * 0.15))
        detail["mood"] = f"{n} check-ins, average mood {round(avg_v * 100)}/100"
    recent = [d for d in st.get("recentDays", []) if d >= s14]
    comp["rhythm"] = min(1.0, len(recent) / 8)                 # ~every other day is full marks
    detail["rhythm"] = f"Active {len(recent)} of the last 14 days"

    entries = [d for _, d in repo.query(user_path(uid, "journal_entries"), where=[("localDay", ">=", s14)])
               if d.get("status") == "saved"]
    jdays = len({e["localDay"] for e in entries})
    comp["reflection"] = min(1.0, jdays / 5)                   # 1-3 day spacing -> ~5 days in 14
    detail["reflection"] = f"Journaled on {jdays} days"

    s7 = _since(tz, 7)
    mins = sum(d.get("minutes", 0) for _, d in repo.query(user_path(uid, "listening_sessions"),
                                                           where=[("startedAt", ">=", s7)]))
    comp["practice"] = min(1.0, mins / 60)
    detail["practice"] = f"{round(mins)} minutes of soundscapes this week"

    plan = active_plan(repo, uid)
    if plan:
        items = [i for d in plan["days"] if d["date"] <= today and d["date"] >= s7 for i in d["items"]]
        if items:
            comp["plan"] = sum(1 for i in items if i.get("done")) / len(items)
            detail["plan"] = f"{sum(1 for i in items if i.get('done'))} of {len(items)} plan steps done"

    used = {k: v for k, v in comp.items() if v is not None}
    wsum = sum(WEIGHTS[k] for k in used)
    score = round(100 * sum(WEIGHTS[k] * v for k, v in used.items()) / wsum) if wsum else 0
    band = next(name for cut, name in BANDS if score >= cut)
    weakest = min(used, key=lambda k: used[k]) if used else "mood"
    lever = next(a for a in core.dataset("selfcare_activities")["activities"] if a["id"] == LEVER[weakest])
    prev = repo.query(user_path(uid, "scores"), where=[("day", "<", today)], order_by="day", desc=True, limit=1)
    delta = score - prev[0][1]["score"] if prev else None
    out = {
        "day": today, "score": score, "band": band, "delta": delta,
        "components": {k: {"value": round(v * 100), "weight": WEIGHTS[k], "detail": detail.get(k)} for k, v in used.items()},
        "insight": _insight(band, weakest, delta),
        "nextStep": {"activityId": lever["id"], "title": lever["title"], "minutes": lever["minutes"],
                     "deeplink": core.resolve_route(lever["route"], lever.get("params"))},
        "rev": _rev(st), "disclaimer": "A wellbeing habit score, not a clinical measure.",
    }
    repo.set(user_path(uid, "scores", today), out)
    return out


def _rev(st: Dict[str, Any]) -> str:
    return f"{st.get('moodRev', 0)}:{st.get('lifetimeCoins', 0)}:{st.get('lastActiveDay')}"


def _insight(band: str, weakest: str, delta: Optional[int]) -> str:
    tip = {"mood": "a quick Mood Scan helps Lucille see how you're really doing",
           "rhythm": "showing up for a minute every couple of days counts",
           "reflection": "a few lines in the Journal goes a long way",
           "practice": "ten minutes of a soundscape would lift this",
           "plan": "one small step from your plan today would help"}[weakest]
    trend = "" if delta is None else (" Up from yesterday." if delta > 0 else (" A little lower than yesterday — that's okay." if delta < 0 else ""))
    return f"{band}.{trend} Next: {tip}."


# ───────────────────────── plans ─────────────────────────

GOALS = {
    "sleep_better": {"label": "Sleep better", "kinds": ["sleep", "breathe", "listen"], "pillars": ["sleep"]},
    "less_stress": {"label": "Feel less stressed", "kinds": ["breathe", "meditate", "listen", "move"], "pillars": ["mind"]},
    "focus": {"label": "Focus", "kinds": ["focus", "listen", "breathe"], "pillars": ["mind"]},
    "lift_mood": {"label": "Lift my mood", "kinds": ["journal", "move", "connect", "listen"], "pillars": ["reflect", "body"]},
    "move_more": {"label": "Move more", "kinds": ["move"], "pillars": ["body"]},
    "connect": {"label": "Feel more connected", "kinds": ["talk", "connect", "journal"], "pillars": ["connect"]},
    "know_myself": {"label": "Understand myself", "kinds": ["journal", "checkin"], "pillars": ["reflect"]},
}
SLOTS = ("morning", "day", "evening")


class PlanProvider:
    name = "base"

    async def build(self, goals: List[str], minutes: int, days: int, mood_word: Optional[str], tz: str) -> List[List[str]]:
        raise NotImplementedError


class CatalogPlanProvider(PlanProvider):
    """Deterministic, free. Rotates through matching activities so days differ."""
    name = "catalog"

    async def build(self, goals, minutes, days, mood_word, tz):
        acts = core.dataset("selfcare_activities")["activities"]
        kinds = {k for g in goals for k in GOALS.get(g, {}).get("kinds", [])} or {"breathe", "journal", "listen"}
        w = (mood_word or "").lower()
        pool = [a for a in acts if a["kind"] in kinds and not a.get("requiresCoins")]
        pool.sort(key=lambda a: (-(w in a["words"]), a["minutes"]))
        evening = [a for a in pool if a.get("evening")] or pool
        morning = [a for a in pool if a.get("morning") or a["energy"] == "up"] or pool
        day_pool = [a for a in pool if not a.get("evening")] or pool
        out = []
        for i in range(days):
            picks = [morning[i % len(morning)], day_pool[(i * 2 + 1) % len(day_pool)], evening[i % len(evening)]]
            budget, chosen = minutes, []
            for a in picks:
                if a["id"] in chosen:
                    continue
                if budget - a["minutes"] >= -3 or not chosen:
                    chosen.append(a["id"]); budget -= a["minutes"]
            out.append(chosen)
        return out


class ClaudePlanProvider(PlanProvider):
    """Claude chooses + orders activity ids from the library. Falls back to catalog on any problem."""
    name = "claude"

    async def build(self, goals, minutes, days, mood_word, tz):
        key = os.getenv("ANTHROPIC_API_KEY")
        fallback = await CatalogPlanProvider().build(goals, minutes, days, mood_word, tz)
        if not key:
            return fallback
        acts = core.dataset("selfcare_activities")["activities"]
        lib = [{k: a[k] for k in ("id", "title", "kind", "minutes", "pillar")} | {"evening": a.get("evening", False)}
               for a in acts if not a.get("requiresCoins")]
        prompt = (f"Build a {days}-day self-care plan for goals {goals}, about {minutes} minutes a day, current mood "
                  f"'{mood_word or 'unknown'}'. Use ONLY ids from this library: {json.dumps(lib)}. Each day: 1-3 ids "
                  f"ordered morning, day, evening. Reply with JSON only: {{\"days\": [[ids...], ...]}}")
        try:
            import httpx
            async with httpx.AsyncClient(timeout=25) as cl:
                r = await cl.post("https://api.anthropic.com/v1/messages",
                                  headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                                           "content-type": "application/json"},
                                  json={"model": cfg().anthropic_model, "max_tokens": 600,
                                        "messages": [{"role": "user", "content": prompt}]})
                r.raise_for_status()
                text = "".join(b.get("text", "") for b in r.json().get("content", []))
                data = json.loads(text[text.find("{"): text.rfind("}") + 1])
            valid = {a["id"] for a in lib}
            plan = [[i for i in d if i in valid][:3] for d in data.get("days", [])][:days]
            return plan if len(plan) == days and all(plan) else fallback
        except Exception as e:
            logger.warning(f"Claude plan provider failed, using catalog: {e}")
            return fallback


def provider(name: Optional[str] = None) -> PlanProvider:
    return ClaudePlanProvider() if (name or cfg().plan_provider) == "claude" else CatalogPlanProvider()


async def create_plan(repo: Repo, uid: str, tz: str, goals: List[str], minutes: int, days: int,
                      provider_name: Optional[str] = None) -> Dict[str, Any]:
    st = core.get_state(repo, uid)
    word = (st.get("lastMood") or {}).get("word")
    prov = provider(provider_name)
    ids = await prov.build(goals, minutes, days, word, tz)
    acts = {a["id"]: a for a in core.dataset("selfcare_activities")["activities"]}
    start = date.fromisoformat(core.local_day(tz))
    pid = core.new_id("plan")
    plan_days = []
    for i, day_ids in enumerate(ids):
        items = []
        for j, aid in enumerate(day_ids):
            a = acts[aid]
            items.append({"itemId": f"d{i + 1}_{j + 1}", "activityId": aid, "title": a["title"], "kind": a["kind"],
                          "minutes": a["minutes"], "slot": SLOTS[min(j, 2)], "done": False,
                          "deeplink": core.resolve_route(a["route"], a.get("params"))})
        plan_days.append({"date": (start + timedelta(days=i)).isoformat(), "items": items})
    for _, old in repo.query(user_path(uid, "plans"), where=[("active", "==", True)]):
        repo.set(user_path(uid, "plans", old["id"]), {"active": False}, merge=True)
    plan = {"id": pid, "goals": goals, "goalLabels": [GOALS[g]["label"] for g in goals if g in GOALS],
            "minutesPerDay": minutes, "provider": prov.name, "active": True, "days": plan_days,
            "createdAt": core.iso(core.now_utc()), "startDay": start.isoformat()}
    repo.set(user_path(uid, "plans", pid), plan)
    return plan


def active_plan(repo: Repo, uid: str) -> Optional[Dict[str, Any]]:
    rows = repo.query(user_path(uid, "plans"), where=[("active", "==", True)], limit=1)
    return rows[0][1] if rows else None


def today_items(plan: Optional[Dict[str, Any]], tz: str) -> List[Dict[str, Any]]:
    if not plan:
        return []
    today = core.local_day(tz)
    return next((d["items"] for d in plan["days"] if d["date"] == today), [])
