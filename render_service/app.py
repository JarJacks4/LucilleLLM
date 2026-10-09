"""
Lucille render service (Cloud Run: lucille-render).

POST /v1/render     X-Api-Key: $RENDER_API_KEY
    The body the main API sends from Compose (escape_api/routers/soundscapes.py::_send_render).
    Returns 202 at once, renders in the background (deploy with --no-cpu-throttling),
    uploads the segment set to GCS and calls callbackUrl with X-Lucille-Secret.
POST /v1/render/sync   same body, renders and returns the result in the response (tests, catalog jobs).
GET  /health

Env: RENDER_API_KEY, LUCILLE_RENDER_CALLBACK_SECRET, AUDIO_BUCKET (e.g. escape-self-care-505618-escape-media),
     AUDIO_PUBLIC_BASE (default https://storage.googleapis.com/<bucket>), RENDER_PREFIX (default audio/compose),
     RENDER_WORKERS (default 2).
"""

from __future__ import annotations

import hmac
import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from render_service.encode import encode_m4a, tmpdir
from render_service.synth import Recipe, render_segment, segment_plan

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("lucille-render")
app = FastAPI(title="Lucille render", version="0.1.0")
_pool = ThreadPoolExecutor(max_workers=int(os.getenv("RENDER_WORKERS", "2")))


class Segments(BaseModel):
    introSec: float = Field(45, ge=5, le=300)
    bodyCount: int = Field(4, ge=1, le=8)
    bodySec: float = Field(150, ge=10, le=600)
    outroSec: float = Field(60, ge=5, le=300)


class Output(BaseModel):
    bitrateKbps: int = 160
    loudnessLufs: float = -18
    truePeakDbtp: float = -1


class RenderIn(BaseModel):
    compositionId: str = Field(..., pattern=r"^[A-Za-z0-9_\-]{3,80}$")
    mode: str = "calm"
    prompt: Optional[str] = None
    recipe: Dict
    segments: Segments = Field(default_factory=Segments)
    output: Output = Field(default_factory=Output)
    callbackUrl: Optional[str] = None
    style: Optional[str] = None
    energy: Optional[float] = None
    droneHz: Optional[float] = None
    destPrefix: Optional[str] = Field(None, description="GCS prefix override, e.g. soundscapes/jazz/jazz_01")


def _check_key(key: Optional[str]) -> None:
    want = os.getenv("RENDER_API_KEY", "")
    if not want or not key or not hmac.compare_digest(want, key):
        raise HTTPException(401, "bad api key")


def _upload(local: Path, dest: str) -> str:
    bucket = os.getenv("AUDIO_BUCKET")
    if not bucket:                                   # local dev: keep the file, return a file URL
        return local.as_uri()
    from google.cloud import storage
    blob = storage.Client().bucket(bucket).blob(dest)
    blob.cache_control = "public, max-age=31536000, immutable"
    blob.upload_from_filename(str(local), content_type="audio/mp4")
    base = os.getenv("AUDIO_PUBLIC_BASE", f"https://storage.googleapis.com/{bucket}").rstrip("/")
    return f"{base}/{dest}"


def render(req: RenderIn, keep_dir: Optional[Path] = None) -> Dict:
    t0 = time.time()
    rec = Recipe.from_api(req.recipe, req.mode, energy=req.energy if req.energy is not None else 0.4,
                          style=req.style, drone_hz=req.droneHz)
    work = keep_dir or tmpdir()
    prefix = (req.destPrefix or f"{os.getenv('RENDER_PREFIX', 'audio/compose')}/{req.compositionId}").strip("/")
    segs: List[Dict] = []
    measured: Dict = {}
    plan = segment_plan(req.segments.introSec, req.segments.bodyCount, req.segments.bodySec, req.segments.outroSec)
    try:
        for role, seconds, fi, fo in plan:
            audio = render_segment(rec, seconds, f"{req.compositionId}:{role}", fi, fo)
            path = work / f"{role}.m4a"
            m = encode_m4a(audio, path, req.output.loudnessLufs, req.output.truePeakDbtp, req.output.bitrateKbps)
            url = _upload(path, f"{prefix}/{role}.m4a")
            segs.append({"role": "body" if role.startswith("body") else role, "name": role,
                         "url": url, "durationSec": m["durationSec"], "bytes": m["bytes"]})
            measured = m
        return {"compositionId": req.compositionId, "status": "ready", "segments": segs,
                "measured": {"loudnessLufs": measured.get("loudnessLufs"), "truePeakDbtp": measured.get("truePeakDbtp"),
                             "tempoBpm": rec.tempo, "key": rec.key, "renderSec": round(time.time() - t0, 1)}}
    finally:
        if keep_dir is None:
            shutil.rmtree(work, ignore_errors=True)


def _callback(url: str, body: Dict) -> None:
    secret = os.getenv("LUCILLE_RENDER_CALLBACK_SECRET", "")
    for attempt in range(3):
        try:
            r = httpx.post(url, json=body, headers={"X-Lucille-Secret": secret}, timeout=20)
            if r.status_code < 500:
                return
        except Exception as e:  # pragma: no cover - network
            log.warning(f"callback attempt {attempt + 1} failed: {e}")
        time.sleep(2 * (attempt + 1))


def _job(req: RenderIn) -> None:
    try:
        result = render(req)
    except Exception as e:
        log.exception("render failed")
        result = {"compositionId": req.compositionId, "status": "failed", "segments": [],
                  "failReason": type(e).__name__}
    if req.callbackUrl:
        _callback(req.callbackUrl, {k: v for k, v in result.items() if k != "compositionId"})


@app.get("/health")
def health():
    return {"ok": True, "ffmpeg": bool(shutil.which("ffmpeg"))}


@app.post("/v1/render", status_code=202)
def render_async(req: RenderIn, x_api_key: Optional[str] = Header(default=None)):
    _check_key(x_api_key)
    if req.callbackUrl:
        _callback(req.callbackUrl, {"status": "rendering"})
    _pool.submit(_job, req)
    return {"accepted": True, "compositionId": req.compositionId}


@app.post("/v1/render/sync")
def render_sync(req: RenderIn, x_api_key: Optional[str] = Header(default=None)):
    _check_key(x_api_key)
    return render(req)
