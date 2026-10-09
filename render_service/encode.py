"""Encode float stereo to AAC-LC .m4a (48 kHz, 160 kb/s) with EBU R128 loudness, and measure it."""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np

from render_service.synth import SR


def encode_m4a(audio: np.ndarray, out_path: Path, lufs: float = -18.0, true_peak: float = -1.0,
               bitrate_k: int = 160) -> Dict[str, float]:
    """audio: float32 (2, n). Two-pass loudnorm so the target is hit accurately. AAC-LC only
    (HE-AAC v2 would rebuild stereo from one channel and erase binaural beats)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.ascontiguousarray(audio.T.astype("<f4")).tobytes()
    base = ["ffmpeg", "-hide_banner", "-nostdin", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "pipe:0"]
    target = f"loudnorm=I={lufs}:TP={true_peak}:LRA=11"
    p1 = subprocess.run(base + ["-af", target + ":print_format=json", "-f", "null", "-"],
                        input=pcm, capture_output=True, check=True)
    m = json.loads(re.findall(r"\{[^{}]*\}", p1.stderr.decode())[-1])
    second = (f"{target}:measured_I={m['input_i']}:measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}"
              f":measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true:print_format=json")
    p2 = subprocess.run(base + ["-af", second, "-ar", str(SR), "-c:a", "aac", "-profile:a", "aac_low",
                                "-b:a", f"{bitrate_k}k", "-movflags", "+faststart", "-y", str(out_path)],
                        input=pcm, capture_output=True, check=True)
    m2 = json.loads(re.findall(r"\{[^{}]*\}", p2.stderr.decode())[-1])
    return {"loudnessLufs": round(float(m2["output_i"]), 1), "truePeakDbtp": round(float(m2["output_tp"]), 1),
            "durationSec": round(audio.shape[1] / SR, 2), "bytes": out_path.stat().st_size}


def tmpdir() -> Path:
    return Path(tempfile.mkdtemp(prefix="lucille_render_"))
