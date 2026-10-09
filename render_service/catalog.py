"""
Pre-render the Soundscapes catalog (escape_api/data/soundscape_catalog.json) to GCS:

    soundscapes/{category}/{trackId}/intro.m4a, body_1..4.m4a, outro.m4a, preview.m4a

    AUDIO_BUCKET=escape-self-care-ai.appspot.com python -m render_service.catalog            # all tracks
    python -m render_service.catalog --only jazz_01 --local out/                              # one, to disk
    python -m render_service.catalog --free-only --preview-only                               # just previews

Each category maps to a style + layers so the 9 categories sound distinct.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from render_service.app import RenderIn, render
from render_service.encode import encode_m4a
from render_service.synth import Recipe, render_segment

CATALOG = Path(__file__).resolve().parents[1] / "escape_api" / "data" / "soundscape_catalog.json"

NATURE_BED = {"rain": "rain_roof", "ocean": "ocean", "forest": "forest", "stream": "stream", "fire": "fire"}

STYLE = {
    "music_meditations": dict(style="bowls", melody="bowls", ambience="soft_air", pulse=None, scale="major_pentatonic", reverb=0.8),
    "vaporwave": dict(style="vaporwave", melody="soft_pads", ambience=None, pulse="soft_kick", scale="lydian", reverb=0.6),
    "jazz": dict(style="jazz", melody="felt_keys", ambience="rain_roof", pulse="brush", scale="dorian", reverb=0.35),
    "nature": dict(style="ambient", melody="none", ambience="forest", pulse=None, scale="major_pentatonic", reverb=0.25),
    "binaural_beats": dict(style="ambient", melody="soft_pads", ambience="soft_air", pulse=None, scale="major_pentatonic", reverb=0.6),
    "brainwave_music": dict(style="ambient", melody="soft_pads", ambience="soft_air", pulse="slow_heartbeat", scale="minor_pentatonic", reverb=0.55, binaural=False),
    "raw_frequencies": dict(style="drone", melody=None, ambience=None, pulse=None, scale="major_pentatonic", reverb=0.4),
    "sleep_ambient": dict(style="ambient", melody="soft_pads", ambience="soft_air", pulse=None, scale="major_pentatonic", reverb=0.85),
    "depression_anxiety": dict(style="ambient", melody="felt_keys", ambience="soft_air", pulse="slow_heartbeat", scale="major_pentatonic", reverb=0.7),
}
LUFS = {"focus": -16, "calm": -18, "sleep": -20, "move": -14}


def recipe_for(track: dict) -> dict:
    st = STYLE[track["category"]]
    mf = track["moodField"]
    amb = st["ambience"]
    if track["category"] == "nature":
        amb = next((v for k, v in NATURE_BED.items() if k in track["title"].lower()), "forest")
    bw = track.get("brainwave")
    return {
        "tempoBpm": track.get("bpm") or 60, "key": track.get("key") or "D", "scale": st["scale"],
        "brightness": round(0.25 + 0.5 * mf["energy"], 2), "reverb": st["reverb"],
        "stereoWidth": round(0.5 + 0.4 * mf["texture"], 2),
        "layers": {"ambience": amb, "melody": st["melody"], "pulse": st["pulse"]},
        "brainwave": bw, "binaural": st.get("binaural", True), "style": st["style"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--category")
    ap.add_argument("--free-only", action="store_true")
    ap.add_argument("--preview-only", action="store_true")
    ap.add_argument("--local", help="write here instead of uploading")
    a = ap.parse_args()
    tracks = json.loads(CATALOG.read_text())["tracks"]
    for t in tracks:
        if a.only and t["id"] not in a.only:
            continue
        if a.category and t["category"] != a.category:
            continue
        if a.free_only and t["premium"]:
            continue
        rec = recipe_for(t)
        prefix = t["audio"]["path"].rstrip("/")
        out_dir = Path(a.local) / prefix if a.local else None
        if a.local:
            os.environ.pop("AUDIO_BUCKET", None)
            out_dir.mkdir(parents=True, exist_ok=True)
        # 30 s preview from its own seed
        r = Recipe.from_api(rec, t["mode"], energy=t["moodField"]["energy"], style=rec["style"], drone_hz=t.get("frequencyHz"))
        prev = render_segment(r, 30, f"{t['id']}:preview", 2.0, 4.0)
        pdir = out_dir or Path("/tmp") / prefix
        m = encode_m4a(prev, pdir / "preview.m4a", LUFS.get(t["mode"], -18))
        if not a.local:
            from render_service.app import _upload
            _upload(pdir / "preview.m4a", f"{prefix}/preview.m4a")
        print(f"{t['id']}: preview {m['loudnessLufs']} LUFS")
        if a.preview_only:
            continue
        req = RenderIn(compositionId=t["id"], mode=t["mode"], recipe=rec, style=rec["style"],
                       energy=t["moodField"]["energy"], droneHz=t.get("frequencyHz"), destPrefix=prefix,
                       output={"loudnessLufs": LUFS.get(t["mode"], -18)})
        res = render(req, keep_dir=out_dir)
        print(f"{t['id']}: {len(res['segments'])} segments in {res['measured']['renderSec']} s")


if __name__ == "__main__":
    main()
