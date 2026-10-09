"""/v1/config, /v1/health, scheduled jobs and webhooks."""

from __future__ import annotations

import hmac
import os
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel

from auth_middleware import require_admin
from escape_api import core
from escape_api.repo import get_repo, user_path
from escape_api.settings import cfg

router = APIRouter(prefix="/v1", tags=["v1 · config & jobs"])

API_VERSION = "1.0.0"


@router.get("/health", summary="Liveness for the v1 routers")
def health():
    return {"ok": True, "version": API_VERSION, "llm": bool(os.getenv("OPENAI_API_KEY")) and cfg().llm_enabled}


@router.get("/config", summary="Feature flags + static config the app reads once at launch (cache 1h)")
def config():
    s = cfg()
    return {
        "version": API_VERSION,
        "features": {"journalV2": True, "moodStatsV2": True, "soundscapesV2": True, "selfCarePlans": True,
                     "compose": bool(s.render_url), "claudePlans": s.plan_provider == "claude",
                     "weather": s.weather_provider != "none"},
        "enforceConsent": s.enforce_consent,
        "assetBaseUrl": s.asset_base_url, "audioBaseUrl": s.audio_base_url,
        "deeplinkScheme": core.dataset("deeplinks")["scheme"],
        "aiDisclosure": core.AI_DISCLOSURE,
        "energyLevelStep": s.energy_level_step,
        "limits": {"reflectPerDay": s.quota_reflect_per_day, "composePerDayFree": s.quota_compose_per_day_free},
    }


class Job(BaseModel):
    dryRun: bool = False


@router.post("/jobs/letters-due", dependencies=[Depends(require_admin)],
             summary="Cloud Scheduler (hourly): deliver due 'letters to future self' as push notifications")
def letters_due(body: Job = Job()):
    repo = get_repo()
    now = core.iso(core.now_utc())
    delivered = 0
    # one collection-group query (needs the index in firestore.indexes.json), not a walk over every user
    for path, l in repo.collection_group("journal_letters", where=[("delivered", "==", False), ("deliverAt", "<=", now)], limit=500):
        uid, lid = path.split("/")[1], path.split("/")[-1]
        delivered += 1
        if body.dryRun:
            continue
        # FlutterFlow push format (ff_user_push_notifications triggers the FF push function)
        repo.set(f"ff_user_push_notifications/{core.new_id('push')}", {
            "notification_title": "A letter from you",
            "notification_text": f"You wrote this {l['deliverInDays']} days ago" + (f", feeling {l['moodWord'].lower()}." if l.get("moodWord") else "."),
            "user_refs": f"/Users/{uid}", "initial_page_name": "JournalHomeV2",
            "parameter_data": f'{{"letterId":"{lid}"}}', "timestamp": core.now_utc(),
        })
        repo.set(user_path(uid, "journal_letters", lid), {"delivered": True, "deliveredAt": now}, merge=True)
    return {"delivered": delivered, "dryRun": body.dryRun}


@router.post("/webhooks/revenuecat", include_in_schema=False)
def revenuecat(event: dict, authorization: Optional[str] = Header(default=None)):
    secret = os.getenv("REVENUECAT_WEBHOOK_SECRET", "")
    if not secret or not authorization or not hmac.compare_digest(authorization, f"Bearer {secret}"):
        raise HTTPException(401, "bad secret")
    ev = event.get("event", {})
    uid = ev.get("app_user_id")
    if not uid:
        return {"ignored": True}
    active = ev.get("type") in ("INITIAL_PURCHASE", "RENEWAL", "UNCANCELLATION", "PRODUCT_CHANGE", "TRANSFER")
    if ev.get("type") in ("EXPIRATION", "BILLING_ISSUE"):
        active = False
    get_repo().set(user_path(uid, "state"), {"isPremium": active, "premiumUpdatedAt": core.iso(core.now_utc())}, merge=True)
    return {"ok": True, "isPremium": active}
