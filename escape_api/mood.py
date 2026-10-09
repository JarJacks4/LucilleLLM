"""
Mood engine: fuse scan inputs into one reading, store it, keep cheap rollups,
and answer 'what is my overall mood this week / month / year' (the single
conglomerate Mood Orb) without scanning every check-in.
"""

from __future__ import annotations

from collections import Counter
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from escape_api import core
from escape_api.repo import Repo, user_path

SOURCES = {"scan", "tap", "voice", "text", "journal", "session_before", "session_after",
           "energy_scan", "checkin", "healthkit"}


def fuse_inputs(
    word: Optional[str] = None,
    valence: Optional[float] = None,
    energy: Optional[float] = None,
    text: Optional[str] = None,
    bpm: Optional[float] = None,
    hrv: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Mix the three Mood Scan inputs in moderation:
      tap (Mood Field / word)  weight 0.5
      voice/text (keywords)    weight 0.3
      pulse (bpm, hrv)         weight 0.2, energy only
    Returns valence, energy, word, family, tone, confidence, used inputs.
    """
    parts_v: List[tuple] = []
    parts_e: List[tuple] = []
    used = []
    w = core.find_word(word)
    if w:
        parts_v.append((w["valence"], 0.5)); parts_e.append((w["energy"], 0.5)); used.append("word")
    if valence is not None and energy is not None:
        parts_v.append((core.clamp01(valence), 0.5)); parts_e.append((core.clamp01(energy), 0.5)); used.append("tap")
    if text:
        tw = core.interpret_text(text)
        if tw:
            parts_v.append((tw["valence"], 0.3)); parts_e.append((tw["energy"], 0.3)); used.append("voice")
    if bpm:
        e_bpm = core.clamp01((float(bpm) - 55) / 50)            # 55 bpm -> 0, 105 bpm -> 1
        if hrv:                                                  # higher HRV reads calmer
            e_bpm = core.clamp01(e_bpm - min(0.2, float(hrv) / 400))
        parts_e.append((e_bpm, 0.2)); used.append("pulse")
    if not parts_v:
        parts_v.append((0.5, 1.0))
    if not parts_e:
        parts_e.append((0.45, 1.0))
    v = sum(x * k for x, k in parts_v) / sum(k for _, k in parts_v)
    e = sum(x * k for x, k in parts_e) / sum(k for _, k in parts_e)
    chosen = w if (w and abs(w["valence"] - v) < 0.2 and abs(w["energy"] - e) < 0.25) else core.nearest_word(v, e)
    tone = core.tone_for(v, e, grateful=chosen["family"] == "tender")
    return {
        "valence": round(v, 3), "energy": round(e, 3),
        "word": chosen["word"], "family": chosen["family"], "tone": tone,
        "confidence": round(min(1.0, 0.35 + 0.2 * len(used)), 2), "inputs": used,
    }


def record_checkin(repo: Repo, uid: str, tz: str, reading: Dict[str, Any], source: str,
                   extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    now = core.now_utc()
    cid = core.new_id("mood")
    day = core.local_day(tz, now)
    doc = {
        "id": cid, "source": source if source in SOURCES else "checkin",
        "valence": reading["valence"], "energy": reading["energy"],
        "word": reading["word"], "family": reading["family"], "tone": reading["tone"],
        "confidence": reading.get("confidence"), "inputs": reading.get("inputs", []),
        "localDay": day, "createdAt": core.iso(now),
        **{k: v for k, v in (extra or {}).items() if v is not None},
    }
    repo.set(user_path(uid, "mood_checkins", cid), doc)
    _rollup(repo, uid, day, reading)
    repo.increment(user_path(uid, "state"), {"moodRev": 1},
                   extra={"lastMood": {k: doc[k] for k in ("id", "word", "tone", "valence", "energy", "createdAt")}})
    return doc


def _rollup(repo: Repo, uid: str, day: str, r: Dict[str, Any], sign: int = 1) -> None:
    deltas = {
        "count": sign, "sumV": sign * r["valence"], "sumE": sign * r["energy"],
        f"words.{r['word']}": sign, f"families.{r['family']}": sign,
        "pleasant": sign * (1 if r["valence"] >= 0.55 else 0),
    }
    repo.increment(user_path(uid, "mood_days", day), deltas, extra={"day": day})
    repo.increment(user_path(uid, "mood_months", day[:7]), deltas, extra={"month": day[:7]})


def delete_checkin(repo: Repo, uid: str, checkin_id: str) -> bool:
    doc = repo.get(user_path(uid, "mood_checkins", checkin_id))
    if not doc:
        return False
    _rollup(repo, uid, doc["localDay"], doc, sign=-1)
    repo.delete(user_path(uid, "mood_checkins", checkin_id))
    repo.increment(user_path(uid, "state"), {"moodRev": 1})
    return True


# ───────────────────────── conglomerate summary ─────────────────────────

_EXPECTED = {"week": 7, "month": 24, "year": 200}
_summary_cache: Dict[str, Dict[str, Any]] = {}


def _days(tz: str, n: int) -> List[str]:
    today = date.fromisoformat(core.local_day(tz))
    return [(today - timedelta(days=i)).isoformat() for i in range(n - 1, -1, -1)]


def _months(tz: str, n: int) -> List[str]:
    today = date.fromisoformat(core.local_day(tz))
    y, m = today.year, today.month
    out = []
    for _ in range(n):
        out.append(f"{y:04d}-{m:02d}")
        m -= 1
        if m == 0:
            y, m = y - 1, 12
    return out[::-1]


def summary(repo: Repo, uid: str, tz: str, period: str = "week", night: bool = False) -> Dict[str, Any]:
    period = period if period in _EXPECTED else "week"
    st = repo.get(user_path(uid, "state")) or {}
    ck = f"{uid}:{period}:{tz}:{st.get('moodRev', 0)}:{core.local_day(tz)}:{night}"
    if ck in _summary_cache:
        return _summary_cache[ck]

    if period == "year":
        keys = _months(tz, 12)
        rows = [repo.get(user_path(uid, "mood_months", k)) or {} for k in keys]   # 12 reads
        buckets = [[r] for r in rows]
        labels = keys
    else:
        n = 7 if period == "week" else 30
        keys = _days(tz, n)
        # one range query instead of n gets
        got = {d["day"]: d for _, d in repo.query(user_path(uid, "mood_days"),
                                                   where=[("day", ">=", keys[0]), ("day", "<=", keys[-1])])}
        rows = [got.get(k, {}) for k in keys]
        size = 1 if period == "week" else 2
        buckets = [rows[i:i + size] for i in range(0, len(rows), size)]
        labels = [keys[i] for i in range(0, len(keys), size)]

    total = sum(r.get("count", 0) for r in rows)
    words: Counter = Counter()
    fams: Counter = Counter()
    for r in rows:
        words.update({k: v for k, v in (r.get("words") or {}).items() if v > 0})
        fams.update({k: v for k, v in (r.get("families") or {}).items() if v > 0})
    active = sum(1 for r in rows if r.get("count", 0) > 0)
    pleasant = sum(r.get("pleasant", 0) for r in rows)

    def avg(rs):
        c = sum(r.get("count", 0) for r in rs)
        return (sum(r.get("sumV", 0) for r in rs) / c, sum(r.get("sumE", 0) for r in rs) / c) if c else (None, None)

    av, ae = avg(rows)
    trend = []
    for lab, b in zip(labels, buckets):
        v, _ = avg(b)
        h = round(v * 100) if v is not None else 0
        trend.append({"label": lab, "h": h, "empty": v is None,
                      "c": "#39D9C1" if h > 55 else ("#8E7CD9" if h and h < 40 else "#4CF6F6")})

    if total == 0:
        tone, word, fam = "mixed", "Not enough yet", "mixed"
        caption = "Check in a few times and your orb will take shape."
    else:
        top_word, top_n = words.most_common(1)[0]
        wmeta = core.find_word(top_word) or core.nearest_word(av, ae)
        word = top_word if top_n / total >= 0.35 else core.nearest_word(av, ae)["word"]
        fam = (core.find_word(word) or wmeta)["family"]
        tone = core.tone_for(av, ae, grateful=fam == "tender")
        caption = _caption(period, trend, word, total)

    expected = _EXPECTED[period]
    size = round(80 + 44 * min(1.0, total / expected))
    mix = []
    for w_, n_ in words.most_common(4):
        meta = core.find_word(w_) or {"family": "mixed"}
        fm = core.family_meta(meta["family"])
        mix.append({"label": w_, "family": meta["family"], "pct": round(100 * n_ / total),
                    "color": fm["color"], "grad": fm["grad"]})

    out = {
        "period": period, "word": word, "family": fam, "tone": tone,
        "avgValence": round(av, 3) if av is not None else None,
        "avgEnergy": round(ae, 3) if ae is not None else None,
        "size": size, "big": round(size * 1.45),
        "caption": caption,
        "sub": f"From {total} check-in{'s' if total != 1 else ''}",
        "entries": total, "activeDays": active,
        "pleasantPct": round(100 * pleasant / total) if total else 0,
        "mix": mix, "trend": trend,
        "families": dict(fams),
        "orb": core.orb_assets(tone, night),
    }
    if len(_summary_cache) > 2000:
        _summary_cache.clear()
    _summary_cache[ck] = out
    return out


def _caption(period: str, trend: List[Dict[str, Any]], word: str, total: int) -> str:
    filled = [t["h"] for t in trend if not t["empty"]]
    span = {"week": "week", "month": "month", "year": "year"}[period]
    if len(filled) < 2:
        return f"Mostly {word.lower()} so far this {span}."
    half = len(filled) // 2
    first, last = sum(filled[:half]) / half, sum(filled[half:]) / (len(filled) - half)
    if last - first > 8:
        return f"A {word.lower()} {span} that has been lifting lately."
    if first - last > 8:
        return f"Mostly {word.lower()} this {span}, with the last stretch feeling heavier."
    return f"A steady, mostly {word.lower()} {span}."


# ───────────────────────── suggestions to improve mood ─────────────────────────

def suggestions_for(word: str, tz: str, limit: int = 3, exclude_kinds: Optional[set] = None) -> List[Dict[str, Any]]:
    """Rank activities from the library for this mood. Every item is allowlist-checked."""
    w = word.lower()
    meta = core.find_word(word) or {"energy": 0.5, "valence": 0.5}
    hour = core.now_utc().astimezone(core.safe_tz(tz)).hour
    evening, morning = hour >= 19 or hour < 4, 5 <= hour < 11
    scored = []
    for a in core.dataset("selfcare_activities")["activities"]:
        if exclude_kinds and a["kind"] in exclude_kinds:
            continue
        s = 3 if w in a["words"] else (1 if "*" in a["words"] else 0)
        if not s:
            continue
        if a.get("evening") and evening:
            s += 1
        if a.get("evening") and morning:
            s -= 2
        if a.get("morning") and morning:
            s += 1
        if meta["energy"] > 0.6 and a["energy"] == "down":
            s += 1
        if meta["energy"] < 0.3 and a["energy"] == "up":
            s += 1
        scored.append((s, a))
    scored.sort(key=lambda x: -x[0])
    out, kinds = [], set()
    for s, a in scored:
        if a["kind"] in kinds:
            continue
        link = core.resolve_route(a["route"], a.get("params"))
        if not link:
            continue
        kinds.add(a["kind"])
        out.append({"activityId": a["id"], "title": a["title"], "kind": a["kind"], "minutes": a["minutes"],
                    "pillar": a["pillar"], "deeplink": link, "score": s,
                    "why": f"Picked for feeling {word.lower()}" + (" this evening" if evening else "")})
        if len(out) >= limit:
            break
    return out
