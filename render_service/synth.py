"""
Lucille render engine v0: procedural soundscapes from a sound recipe.

Takes the deterministic recipe the API builds (tempo, key, scale, brightness, reverb,
stereo width, layers, brainwave) and renders a segment set: intro, N bodies, outro,
all in the same key and tempo so the app can shuffle and crossfade them forever.

Pure numpy/scipy, seeded by the composition id, so the same request renders the same
audio. Not a neural model: it is a reliable, cheap baseline the ML team can replace
layer by layer (swap a function, keep the contract).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal

SR = 48000
NOTE = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "Gb": 6,
        "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}
SCALES = {
    "major_pentatonic": [0, 2, 4, 7, 9],
    "minor_pentatonic": [0, 3, 5, 7, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
}


def midi_hz(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


@dataclass
class Recipe:
    tempo: float = 60
    key: str = "D"
    scale: str = "major_pentatonic"
    brightness: float = 0.4      # 0 dark .. 1 bright
    reverb: float = 0.6          # 0 dry .. 1 huge
    width: float = 0.7           # stereo width 0..1
    ambience: Optional[str] = "soft_air"
    melody: Optional[str] = "soft_pads"
    pulse: Optional[str] = None
    brainwave_hz: Optional[float] = None
    binaural: bool = True        # True = L/R beat (headphones); False = isochronic tremolo
    drone_hz: Optional[float] = None
    style: str = "ambient"       # ambient | jazz | vaporwave | bowls | drone
    energy: float = 0.4

    @staticmethod
    def from_api(r: Dict, mode: str = "calm", energy: float = 0.4, style: Optional[str] = None,
                 drone_hz: Optional[float] = None) -> "Recipe":
        layers = r.get("layers") or {}
        bw = r.get("brainwave") or {}
        return Recipe(
            tempo=float(r.get("tempoBpm") or 60), key=r.get("key") or "D",
            scale=r.get("scale") or "major_pentatonic", brightness=float(r.get("brightness", 0.4)),
            reverb=float(r.get("reverb", 0.6)), width=float(r.get("stereoWidth", 0.7)),
            ambience=layers.get("ambience"), melody=layers.get("melody"), pulse=layers.get("pulse"),
            brainwave_hz=float(bw["hz"]) if bw.get("hz") else None,
            binaural=r.get("binaural", True), drone_hz=drone_hz, style=style or r.get("style", "ambient"),
            energy=energy,
        )


def _rng(seed: str) -> np.random.Generator:
    return np.random.default_rng(int(hashlib.sha256(seed.encode()).hexdigest()[:16], 16))


def _lowpass(x: np.ndarray, hz: float, order: int = 2) -> np.ndarray:
    sos = signal.butter(order, min(hz, SR * 0.45), "low", fs=SR, output="sos")
    return signal.sosfilt(sos, x, axis=-1)


def _bandpass(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    sos = signal.butter(2, [lo, min(hi, SR * 0.45)], "band", fs=SR, output="sos")
    return signal.sosfilt(sos, x, axis=-1)


def _pink(n: int, rng) -> np.ndarray:
    w = rng.standard_normal(n)
    f = np.fft.rfft(w)
    k = np.arange(len(f))
    k[0] = 1
    return np.fft.irfft(f / np.sqrt(k), n)


def _brown(n: int, rng) -> np.ndarray:
    b = np.cumsum(rng.standard_normal(n))
    b -= signal.sosfilt(signal.butter(1, 8, "low", fs=SR, output="sos"), b)   # remove drift
    return b / (np.max(np.abs(b)) + 1e-9)


def _norm(x: np.ndarray, peak: float = 1.0) -> np.ndarray:
    m = np.max(np.abs(x))
    return x * (peak / m) if m > 0 else x


def _env_adsr(n: int, a: float, r: float) -> np.ndarray:
    e = np.ones(n)
    na = min(int(a * SR), n // 2)
    nr = min(int(r * SR), n - na)
    if na:
        e[:na] = np.linspace(0, 1, na)
    if nr:
        e[-nr:] *= np.linspace(1, 0, nr)
    return e


# ───────────────────────── layers ─────────────────────────

def pads(n: int, rec: Recipe, rng, root_midi: int) -> np.ndarray:
    """Slow evolving chords from the scale. Returns stereo (2, n)."""
    scale = SCALES.get(rec.scale, SCALES["major_pentatonic"])
    out = np.zeros((2, n))
    t = np.arange(n) / SR
    change = 18 + 14 * (1 - rec.energy)                  # seconds per chord; slow evolution
    starts = np.arange(0, n / SR, change)
    detune = 0.004 if rec.style != "vaporwave" else 0.012
    for i, st in enumerate(starts):
        a, b = int(st * SR), min(n, int((st + change + 6) * SR))   # overlap 6 s
        seg = t[: b - a]
        degree = rng.integers(0, len(scale))
        chord = [scale[degree], scale[(degree + 2) % len(scale)] + 12 * ((degree + 2) // len(scale)),
                 scale[(degree + 4) % len(scale)] + 12 * ((degree + 4) // len(scale)), 12]
        if rec.style == "jazz":
            chord.append(scale[(degree + 6) % len(scale)] + 12)
        voice = np.zeros_like(seg)
        for j, iv in enumerate(chord):
            f = midi_hz(root_midi + iv)
            for d in (-detune, detune):
                ph = rng.uniform(0, 2 * np.pi)
                voice += np.sin(2 * np.pi * f * (1 + d) * seg + ph) * (0.6 if j else 1.0)
                voice += 0.18 * np.sin(2 * np.pi * 2 * f * (1 + d) * seg + ph)      # soft 2nd harmonic
        env = _env_adsr(len(seg), 5.5, 5.5)
        lfo = 0.85 + 0.15 * np.sin(2 * np.pi * rng.uniform(0.03, 0.08) * seg)
        pan = rng.uniform(-0.4, 0.4)
        out[0, a:b] += voice * env * lfo * (1 - pan) / 2
        out[1, a:b] += voice * env * lfo * (1 + pan) / 2
    if rec.style == "vaporwave":                          # tape wobble + darker
        wob = 1 + 0.003 * np.sin(2 * np.pi * 0.4 * t)
        idx = np.clip(np.cumsum(wob) - 1, 0, n - 1)
        out = np.stack([np.interp(idx, np.arange(n), ch) for ch in out])
    return _lowpass(out, 900 + 5000 * rec.brightness)


def bells(n: int, rec: Recipe, rng, root_midi: int) -> np.ndarray:
    """Sparse soft melody notes (felt keys / bowls). Stereo."""
    scale = SCALES.get(rec.scale, SCALES["major_pentatonic"])
    out = np.zeros((2, n))
    beat = 60.0 / rec.tempo
    prob = 0.12 + 0.35 * rec.energy
    decay = 6.0 if rec.style == "bowls" else 2.2
    partials = [(1, 1.0), (2.76, 0.35), (5.4, 0.12)] if rec.style == "bowls" else [(1, 1.0), (2, 0.25), (3, 0.08)]
    ln = int(decay * 2 * SR)
    tt = np.arange(ln) / SR
    for k in range(int(n / SR / beat)):
        if rng.random() > prob:
            continue
        f = midi_hz(root_midi + 12 + scale[rng.integers(0, len(scale))] + (12 if rng.random() < 0.25 else 0))
        note = sum(a * np.sin(2 * np.pi * f * p * tt) * np.exp(-tt * (1 + p) / decay) for p, a in partials)
        note *= np.minimum(1, tt / 0.01)
        a = int(k * beat * SR + rng.uniform(-0.03, 0.03) * SR)
        if a < 0:
            continue
        b = min(n, a + ln)
        pan = rng.uniform(-0.6, 0.6)
        vel = rng.uniform(0.4, 0.9)
        out[0, a:b] += note[: b - a] * vel * (1 - pan) / 2
        out[1, a:b] += note[: b - a] * vel * (1 + pan) / 2
    return _lowpass(out, 1500 + 7000 * rec.brightness)


def ambience(n: int, kind: Optional[str], rng) -> np.ndarray:
    if not kind or kind == "none":
        return np.zeros((2, n))
    if kind in ("rain_roof", "rain"):
        bed = np.stack([_lowpass(_brown(n, rng) * 0.6 + _pink(n, rng) * 0.02, 2500) for _ in range(2)])
        drops = np.zeros((2, n))
        count = int(n / SR * 35)
        for _ in range(count):
            p, ch = rng.integers(0, n - 400), rng.integers(0, 2)
            drops[ch, p:p + 400] += rng.standard_normal(400) * np.exp(-np.arange(400) / 60) * rng.uniform(0.1, 0.5)
        return _norm(bed) * 0.8 + _norm(_bandpass(drops, 1500, 9000)) * 0.35
    if kind == "ocean":
        t = np.arange(n) / SR
        swell = 0.55 + 0.45 * np.sin(2 * np.pi * t / 9.0) ** 2
        bed = np.stack([_lowpass(_pink(n, rng), 1200) for _ in range(2)])
        return _norm(bed * swell)
    if kind in ("forest", "soft_air"):
        bed = np.stack([_lowpass(_pink(n, rng), 1800 if kind == "soft_air" else 3500) for _ in range(2)])
        bed = _norm(bed) * (0.6 if kind == "soft_air" else 0.7)
        if kind == "forest":                               # sparse bird chirps
            t = np.arange(int(0.18 * SR)) / SR
            for _ in range(int(n / SR / 4)):
                p, ch = rng.integers(0, n - len(t)), rng.integers(0, 2)
                f0 = rng.uniform(2500, 4200)
                chirp = np.sin(2 * np.pi * (f0 + 1800 * t / t[-1]) * t) * np.hanning(len(t)) * rng.uniform(0.05, 0.15)
                bed[ch, p:p + len(t)] += chirp
        return bed
    if kind == "stream":
        return _norm(np.stack([_bandpass(_pink(n, rng), 400, 6000) for _ in range(2)])) * 0.7
    if kind in ("fire", "fireplace"):
        bed = np.stack([_lowpass(_brown(n, rng), 600) for _ in range(2)]) * 0.6
        crackle = np.zeros((2, n))
        for _ in range(int(n / SR * 8)):
            p, ch = rng.integers(0, n - 200), rng.integers(0, 2)
            crackle[ch, p:p + 200] += rng.standard_normal(200) * np.exp(-np.arange(200) / 25)
        return _norm(bed) + _norm(_bandpass(crackle, 2000, 12000)) * 0.4
    if kind == "glass_wind":
        t = np.arange(n) / SR
        sweep = 0.5 + 0.5 * np.sin(2 * np.pi * t / 14)
        return _norm(np.stack([_bandpass(_pink(n, rng), 2500, 7000) for _ in range(2)]) * sweep) * 0.5
    return np.zeros((2, n))


def pulse(n: int, rec: Recipe, rng) -> np.ndarray:
    if not rec.pulse:
        return np.zeros((2, n))
    beat = 60.0 / rec.tempo
    out = np.zeros(n)
    hit_len = int(0.35 * SR)
    tt = np.arange(hit_len) / SR
    kick = np.sin(2 * np.pi * (55 + 60 * np.exp(-tt * 25)) * tt) * np.exp(-tt * 9)
    for k in range(int(n / SR / beat)):
        a = int(k * beat * SR)
        if rec.pulse == "slow_heartbeat":
            if k % 2:
                continue
            for off, g in ((0.0, 0.8), (0.28, 0.55)):
                s = a + int(off * SR)
                b = min(n, s + hit_len)
                if s < n:
                    out[s:b] += kick[: b - s] * g
        elif rec.pulse == "brush" and k % 1 == 0:
            ln = int(0.12 * SR)
            if a + ln < n:
                out[a:a + ln] += _bandpass(rng.standard_normal(ln), 3000, 9000) * np.exp(-np.arange(ln) / (0.04 * SR)) * 0.4
        else:
            b = min(n, a + hit_len)
            out[a:b] += kick[: b - a] * 0.6
    return np.stack([out, out])


def brainwave(n: int, rec: Recipe) -> np.ndarray:
    """Binaural beat: left carrier f, right f + hz (needs stereo + headphones; AAC-LC keeps it)."""
    if not rec.brainwave_hz:
        return np.zeros((2, n))
    t = np.arange(n) / SR
    carrier = 140.0 if rec.brainwave_hz < 4 else (180.0 if rec.brainwave_hz < 8 else 220.0)
    if rec.binaural:
        left = np.sin(2 * np.pi * carrier * t)
        right = np.sin(2 * np.pi * (carrier + rec.brainwave_hz) * t)
        return np.stack([left, right])
    trem = 0.5 + 0.5 * np.sin(2 * np.pi * rec.brainwave_hz * t)        # isochronic
    tone = np.sin(2 * np.pi * carrier * t) * trem
    return np.stack([tone, tone])


def drone(n: int, hz: Optional[float]) -> np.ndarray:
    if not hz:
        return np.zeros((2, n))
    t = np.arange(n) / SR
    trem = 0.92 + 0.08 * np.sin(2 * np.pi * 0.07 * t)
    l = np.sin(2 * np.pi * hz * t) * trem
    r = np.sin(2 * np.pi * hz * t + 0.6) * trem
    return np.stack([l, r])


def reverb(x: np.ndarray, amount: float, rng) -> np.ndarray:
    if amount <= 0.02:
        return x
    seconds = 1.2 + 3.5 * amount
    ln = int(seconds * SR)
    t = np.arange(ln) / SR
    ir = np.stack([rng.standard_normal(ln) * np.exp(-t * 6.9 / seconds) for _ in range(2)])
    ir = _lowpass(ir, 6000) / np.sqrt(np.sum(ir ** 2, axis=1, keepdims=True))
    wet = np.stack([signal.fftconvolve(x[c], ir[c])[: x.shape[1]] for c in range(2)])
    return x * (1 - 0.5 * amount) + _norm(wet, np.max(np.abs(x)) + 1e-9) * (0.7 * amount)


def widen(x: np.ndarray, width: float) -> np.ndarray:
    mid, side = (x[0] + x[1]) / 2, (x[0] - x[1]) / 2
    side *= 0.4 + 1.2 * width
    return np.stack([mid + side, mid - side])


# ───────────────────────── segment set ─────────────────────────

LAYER_GAIN = {"pads": 0.55, "bells": 0.32, "ambience": 0.42, "pulse": 0.35, "brainwave": 0.08, "drone": 0.5}


def render_segment(rec: Recipe, seconds: float, seed: str, fade_in: float = 0.05, fade_out: float = 0.05) -> np.ndarray:
    """One stereo float32 segment in [-1, 1] (loudness is set later by the encoder)."""
    rng = _rng(seed)
    n = int(seconds * SR)
    root = 48 + NOTE.get(rec.key, 2)                       # around C3
    mix = np.zeros((2, n))
    if rec.style == "drone":
        mix += LAYER_GAIN["drone"] * _norm(drone(n, rec.drone_hz))
        mix += 0.12 * _norm(pads(n, rec, rng, root)) if rec.melody else 0
    else:
        if rec.melody != "none":
            mix += LAYER_GAIN["pads"] * _norm(pads(n, rec, rng, root))
            if rec.melody in ("felt_keys", "bells", "bowls") or rec.style in ("jazz", "bowls"):
                mix += LAYER_GAIN["bells"] * _norm(bells(n, rec, rng, root))
        mix += LAYER_GAIN["ambience"] * _norm(ambience(n, rec.ambience, rng))
        mix += (0.1 if rec.pulse == "brush" else LAYER_GAIN["pulse"]) * _norm(_lowpass(pulse(n, rec, rng), 7000))
    mix = reverb(mix, rec.reverb, rng)
    mix = widen(mix, rec.width)
    mix += LAYER_GAIN["brainwave"] * brainwave(n, rec)    # added after widening so L/R stays exact
    mix = _norm(mix, 0.89)
    mix *= _env_adsr(n, fade_in, fade_out)
    return mix.astype(np.float32)


def segment_plan(intro: float = 45, bodies: int = 4, body: float = 150, outro: float = 60) -> List[Tuple[str, float, float, float]]:
    """(role, seconds, fade_in, fade_out). Bodies have tiny fades; the player crossfades 4 s."""
    plan = [("intro", intro, 6.0, 0.05)]
    plan += [(f"body_{i + 1}", body, 0.05, 0.05) for i in range(bodies)]
    plan.append(("outro", outro, 0.05, 20.0))
    return plan
