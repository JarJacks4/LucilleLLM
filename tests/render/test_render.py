"""Render service: short renders, binaural integrity, loudness, auth, callback, catalog styles."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
pytest.importorskip("scipy")
pytestmark = pytest.mark.skipif(subprocess.run(["which", "ffmpeg"], capture_output=True).returncode != 0,
                                reason="ffmpeg not installed")

from render_service import synth  # noqa: E402
from render_service.app import RenderIn, app, render  # noqa: E402
from render_service.catalog import STYLE, recipe_for  # noqa: E402

SHORT = {"introSec": 6, "bodyCount": 2, "bodySec": 10, "outroSec": 6}


def decode(path):
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "2", "-ar", "48000", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(pcm, "<f4").reshape(-1, 2).T


def test_segment_set_binaural_and_loudness(tmp_path):
    rec = {"tempoBpm": 50, "key": "D", "brightness": 0.3, "reverb": 0.6, "stereoWidth": 0.7,
           "layers": {"ambience": "rain_roof", "melody": "soft_pads", "pulse": "slow_heartbeat"},
           "brainwave": {"type": "theta", "hz": 6}}
    out = render(RenderIn(compositionId="cmp_t1", mode="sleep", recipe=rec, segments=SHORT,
                          output={"loudnessLufs": -20}), keep_dir=tmp_path)
    assert out["status"] == "ready" and [s["name"] for s in out["segments"]] == ["intro", "body_1", "body_2", "outro"]
    assert abs(out["measured"]["loudnessLufs"] + 20) <= 1.0 and out["measured"]["truePeakDbtp"] <= -0.9
    x = decode(tmp_path / "body_1.m4a")
    f = np.fft.rfftfreq(x.shape[1], 1 / 48000)
    band = (f > 170) & (f < 195)
    peaks = [f[band][np.argmax(np.abs(np.fft.rfft(x[c]))[band])] for c in (0, 1)]
    assert abs((peaks[1] - peaks[0]) - 6) < 0.3          # 6 Hz theta beat survives AAC-LC


def test_deterministic_by_seed():
    r = synth.Recipe(tempo=60, key="C")
    a = synth.render_segment(r, 3, "same")
    b = synth.render_segment(r, 3, "same")
    c = synth.render_segment(r, 3, "other")
    assert np.allclose(a, b) and not np.allclose(a, c)


@pytest.mark.parametrize("cat", list(STYLE))
def test_every_category_renders_sound(cat):
    import json
    tracks = json.loads((Path(__file__).resolve().parents[2] / "escape_api/data/soundscape_catalog.json").read_text())["tracks"]
    t = next(t for t in tracks if t["category"] == cat)
    rec = recipe_for(t)
    r = synth.Recipe.from_api(rec, t["mode"], energy=t["moodField"]["energy"], style=rec["style"], drone_hz=t.get("frequencyHz"))
    a = synth.render_segment(r, 4, t["id"])
    assert a.shape == (2, 4 * synth.SR) and np.isfinite(a).all() and np.sqrt(np.mean(a ** 2)) > 0.02


def test_api_key_and_callback(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("RENDER_API_KEY", "k")
    monkeypatch.delenv("AUDIO_BUCKET", raising=False)
    c = TestClient(app)
    body = {"compositionId": "cmp_t2", "recipe": {"tempoBpm": 60}, "segments": SHORT}
    assert c.post("/v1/render/sync", json=body).status_code == 401
    r = c.post("/v1/render/sync", json=body, headers={"X-Api-Key": "k"})
    assert r.status_code == 200 and r.json()["status"] == "ready"
    sent = []
    import render_service.app as appmod
    monkeypatch.setattr(appmod, "_callback", lambda url, b: sent.append((url, b)))
    appmod._job(RenderIn(**body, callbackUrl="https://api/x"))
    assert sent and sent[-1][1]["status"] == "ready" and len(sent[-1][1]["segments"]) == 4
