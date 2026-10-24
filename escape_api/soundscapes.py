"""Soundscapes logic: catalog, Lucille Picks, sound recipe, Inner Weather. Catalog is static (zero DB reads)."""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Dict, List, Optional, Tuple

from escape_api import core
from escape_api.settings import cfg

logger = logging.getLogger(__name__)

BRAINWAVE_HZ = {"delta": 2, "theta": 6, "alpha": 10, "beta": 16}


def catalog() -> Dict[str, Any]:
    return core.dataset("soundscape_catalog")


def categories() -> List[Dict[str, Any]]:
    tracks = catalog()["tracks"]
    return [dict(c, trackCount=sum(1 for t in tracks if t["category"] == c["id"]),
                 seeAll=core.resolve_route("AISoundscapesFINAL", {"category": c["id"]}))
            for c in catalog()["categories"]]


def track_out(t: Dict[str, Any], night: bool = False) -> Dict[str, Any]:
    base = cfg().audio_base_url.rstrip("/") + "/" + t["audio"]["path"]
    seg = t["audio"]["segments"]
    mf = t["moodField"]
    return {
        **{k: t[k] for k in ("id", "title", "category", "mode", "moodField", "bpm", "key", "brainwave",
                             "frequencyHz", "durationSec", "premium", "composedBy", "headphones")},
        "aiLabel": "Composed by Lucille (AI)",
        "audio": {"intro": base + seg["intro"], "bodies": [base + b for b in seg["bodies"]],
                  "outro": base + seg["outro"], "preview": base + t["audio"]["preview"],
                  "codec": "aac-lc", "container": "m4a", "crossfadeSec": 4},
        "visual": core.soundscape_loop(t["mode"], mf["energy"], mf["texture"], night),
    }


def find_track(track_id: str) -> Optional[Dict[str, Any]]:
    return next((t for t in catalog()["tracks"] if t["id"] == track_id), None)


def search(category: Optional[str] = None, mode: Optional[str] = None, q: Optional[str] = None,
           word: Optional[str] = None) -> List[Dict[str, Any]]:
    out = catalog()["tracks"]
    if category:
        out = [t for t in out if t["category"] == category]
    if mode:
        out = [t for t in out if t["mode"] == mode]
    if q:
        ql = q.lower()
        out = [t for t in out if ql in t["title"].lower() or ql in t["category"] or any(ql in w for w in t["targetWords"])]
    if word:
        wl = word.lower()
        out = sorted(out, key=lambda t: wl not in t["targetWords"])
    return out


def auto_mode(tz: str, mood_word: Optional[str], night: bool) -> Tuple[str, str]:
    """Lucille Picks: choose the mode and say why."""
    phase = core.day_phase(tz)
    w = core.find_word(mood_word) if mood_word else None
    fam = w["family"] if w else None
    if night or phase in ("Night Drift", "Deep Night"):
        return "sleep", f"It's {phase.lower()}, so I picked something to help you wind down."
    if fam in ("restless", "heavy"):
        return "calm", f"You checked in {mood_word.lower()}, so I picked something steadying."
    if fam in ("energized",) and phase in ("Golden Hour Unwind",):
        return "calm", "Good energy, late in the day — something to land it gently."
    return "focus", f"It's {phase.lower()}" + (f" and you're {mood_word.lower()}" if mood_word else "") + ", so I picked Focus."


def pick(mode: str, energy: Optional[float], texture: Optional[float], word: Optional[str],
         premium: bool) -> Dict[str, Any]:
    pool = [t for t in catalog()["tracks"] if t["mode"] == mode and (premium or not t["premium"])] or \
           [t for t in catalog()["tracks"] if t["mode"] == mode]
    e = 0.5 if energy is None else energy
    x = 0.5 if texture is None else texture
    wl = (word or "").lower()
    return min(pool, key=lambda t: (t["moodField"]["energy"] - e) ** 2 + (t["moodField"]["texture"] - x) ** 2
               - (0.08 if wl and wl in t["targetWords"] else 0))


def recipe(mode: str, energy: float, texture: float, weather: Optional[str], phase: str,
           brainwave: Optional[str]) -> Dict[str, Any]:
    """Deterministic sound recipe (Soundscapes guide, workflow 1). Cheaper and steadier than asking an LLM."""
    lo, hi = catalog()["modes"].get(mode, {}).get("bpm", [55, 75])
    if phase in ("Night Drift", "Deep Night"):
        energy = min(energy, 0.35)
    tempo = round(lo + (hi - lo) * energy)
    layers = {"ambience": "soft_air", "melody": "soft_pads" if texture > 0.5 else "felt_keys",
              "pulse": "slow_heartbeat" if energy < 0.4 else "soft_kick"}
    brightness = 0.3 + 0.5 * energy
    if weather == "rain":
        layers["ambience"], brightness = "rain_roof", brightness - 0.1
    elif weather == "snow":
        layers["ambience"], brightness = "glass_wind", brightness - 0.05
    elif weather == "clear":
        brightness += 0.1
    bw = brainwave or {"sleep": "delta", "calm": "alpha", "focus": "beta"}.get(mode)
    return {
        "tempoBpm": tempo, "key": random.Random(f"{mode}{round(energy, 1)}{round(texture, 1)}").choice(["C", "D", "E", "F", "G", "A"]),
        "scale": "major_pentatonic", "evolution": "slow",
        "brightness": round(max(0, min(1, brightness)), 2), "reverb": round(0.3 + 0.5 * texture, 2),
        "stereoWidth": round(0.5 + 0.4 * texture, 2), "layers": layers,
        "brainwave": {"type": bw, "hz": BRAINWAVE_HZ.get(bw)} if bw else None,
        "loudnessLufs": {"focus": -16, "calm": -18, "sleep": -20, "move": -14}.get(mode, -18),
    }


# ───────────────────────── Inner Weather (cached 15 min per ~10 km cell) ─────────────────────────

_weather_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}


def _bucket(code: Optional[int], temp: Optional[float]) -> str:
    if code is None:
        return "unknown"
    if code in (71, 73, 75, 77, 85, 86):
        return "snow"
    if code >= 51:
        return "rain"
    if code >= 2:
        return "cloudy"
    if temp is not None and temp >= 30:
        return "heat"
    return "clear"


def weather(lat: float, lon: float) -> Dict[str, Any]:
    key = f"{round(lat, 1)},{round(lon, 1)}"
    hit = _weather_cache.get(key)
    if hit and time.time() - hit[0] < 900:
        return hit[1]
    out: Dict[str, Any] = {"available": False, "bucket": "unknown"}
    if cfg().weather_provider == "open-meteo":
        try:
            import httpx
            r = httpx.get("https://api.open-meteo.com/v1/forecast",
                          params={"latitude": round(lat, 2), "longitude": round(lon, 2),
                                  "current": "temperature_2m,weather_code", "timezone": "auto"}, timeout=4)
            cur = r.json().get("current", {})
            t, code = cur.get("temperature_2m"), cur.get("weather_code")
            out = {"available": True, "tempC": t, "code": code, "bucket": _bucket(code, t), "provider": "open-meteo"}
        except Exception as e:
            logger.info(f"weather unavailable: {e}")
    if len(_weather_cache) > 5000:
        _weather_cache.clear()
    _weather_cache[key] = (time.time(), out)
    return out
