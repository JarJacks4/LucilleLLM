"""
/v1/privacy — GDPR / state privacy + AI-companion law endpoints.

Art. 7  consent (granular, withdrawable, with history)       PUT  /v1/privacy/consents
Art. 9  explicit consent before mood/journal (health) data    enforced in routers via require_consent
Art. 13 transparency + AI disclosure                          GET  /v1/privacy/disclosure
Art. 15/20 access + portability                               GET  /v1/privacy/export
Art. 16 rectification                                         PATCH journal entries, DELETE check-ins
Art. 17 erasure (scoped or full account)                      DELETE /v1/privacy/data, POST /v1/privacy/delete-account
Art. 18 restriction of processing                             POST /v1/privacy/restrict
Art. 21 objection to profiling/personalisation                consents.personalization = false
"""

from __future__ import annotations

import logging
from typing import Dict, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from escape_api import core
from escape_api.deps import PURPOSES, Caller, caller
from escape_api.repo import ROOT, user_path

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/privacy", tags=["v1 · privacy (GDPR)"])

POLICY_VERSION = "2026-10"

SCOPES = {
    "mood": ["mood_checkins", "mood_days", "mood_months"],
    "journal": ["journal_entries", "journal_letters", "weekly"],
    "soundscapes": ["listening_sessions", "library", "compositions"],
    "plans": ["plans", "scores"],
}


class ConsentsIn(BaseModel):
    flags: Dict[str, bool] = Field(..., description=f"Any of: {', '.join(PURPOSES)}")
    policyVersion: str = POLICY_VERSION


class DeleteAccountIn(BaseModel):
    confirm: Literal["DELETE"]
    alsoDeleteAuthAccount: bool = True
    alsoDeleteLegacyData: bool = True


class RestrictIn(BaseModel):
    restricted: bool


def _ip(request: Request) -> str:
    f = request.headers.get("x-forwarded-for")
    return f.split(",")[0].strip() if f else (request.client.host if request.client else "")


@router.get("/purposes", summary="What each consent covers (show in onboarding + Settings)")
def purposes():
    return {"policyVersion": POLICY_VERSION, "purposes": [{"key": k, "label": v} for k, v in PURPOSES.items()],
            "required": [], "note": "Nothing is required to use Escape; features that need a purpose ask for it at the moment they're used."}


@router.get("/consents", summary="Current consent flags + history")
def get_consents(c: Caller = Depends(caller)):
    d = c.repo.get(user_path(c.uid, "consents")) or {}
    return {"flags": {k: bool(d.get("flags", {}).get(k)) for k in PURPOSES},
            "restricted": bool(d.get("restricted")), "policyVersion": d.get("policyVersion"),
            "history": d.get("history", [])[-20:]}


@router.put("/consents", summary="Grant or withdraw consents (withdrawal is as easy as granting)")
def put_consents(body: ConsentsIn, request: Request, c: Caller = Depends(caller)):
    unknown = set(body.flags) - set(PURPOSES)
    if unknown:
        raise HTTPException(422, f"Unknown purposes: {sorted(unknown)}")
    d = c.repo.get(user_path(c.uid, "consents")) or {}
    flags = {**d.get("flags", {}), **body.flags}
    hist = d.get("history", [])
    hist.append({"at": core.iso(core.now_utc()), "changes": body.flags, "policyVersion": body.policyVersion,
                 "ip": _ip(request)[:45]})
    c.repo.set(user_path(c.uid, "consents"), {"flags": flags, "policyVersion": body.policyVersion,
                                              "history": hist[-100:], "updatedAt": core.iso(core.now_utc())}, merge=True)
    return get_consents(c)


@router.post("/restrict", summary="Art. 18: pause all optional processing (AI reflection, personalisation, analytics)")
def restrict(body: RestrictIn, c: Caller = Depends(caller)):
    d = c.repo.get(user_path(c.uid, "consents")) or {}
    flags = d.get("flags", {})
    if body.restricted:
        for k in ("ai_reflection", "personalization", "analytics", "location_weather", "health_data"):
            flags[k] = False
    c.repo.set(user_path(c.uid, "consents"), {"flags": flags, "restricted": body.restricted}, merge=True)
    return get_consents(c)


@router.get("/disclosure", summary="AI disclosure, crisis resources, what Escape stores (show on first chat + every few hours)")
def disclosure():
    return {
        "ai": core.AI_DISCLOSURE,
        "repeatEveryMinutes": 180, "repeatEveryMinutesMinor": 60,
        "notTherapy": "Escape and Lucille provide self-care support, not therapy, diagnosis or treatment.",
        "crisis": core.safety_check("suicide")["resources"] or [{"name": "988 Suicide & Crisis Lifeline", "phone": "988"}],
        "stores": {"mood": "Check-ins you save (word, Mood Field position, optional pulse number). Camera frames never leave your phone.",
                   "journal": "Entries, prompts, Lucille's reflections. Voice audio only if you keep it.",
                   "soundscapes": "What you listened to and for how long.",
                   "ai": "If you allow AI reflection, the entry text is sent to our AI provider to write a reflection, and not used to train models."},
        "policyVersion": POLICY_VERSION,
    }


@router.get("/export", summary="Art. 15/20: everything we hold for you as JSON (v1 data + legacy Lucille data)")
def export(request: Request, c: Caller = Depends(caller), includeLegacy: bool = True):
    data = {"exportedAt": core.iso(core.now_utc()), "uid": c.uid, "format": "escape-export-v1",
            "escape": c.repo.dump_tree(f"{ROOT}/{c.uid}")}
    if includeLegacy:
        try:
            from compliance_service import get_compliance_service
            data["lucilleLegacy"] = get_compliance_service().export_user_data(c.uid, ip_address=_ip(request)).model_dump(mode="json")
        except Exception as e:
            data["lucilleLegacy"] = {"error": f"legacy export unavailable: {type(e).__name__}"}
    return JSONResponse(data, headers={"Content-Disposition": f'attachment; filename="escape-export-{c.uid[:8]}.json"'})


@router.delete("/data", summary="Art. 17: erase one area (mood, journal, soundscapes, plans) or all v1 data")
def delete_scope(scope: Literal["mood", "journal", "soundscapes", "plans", "all"], c: Caller = Depends(caller)):
    n = 0
    if scope == "all":
        consents = c.repo.get(user_path(c.uid, "consents"))
        n = c.repo.delete_tree(f"{ROOT}/{c.uid}")
        # new revision so no instance serves a cached orb/score built from erased data
        c.repo.set(user_path(c.uid, "state"), {"moodRev": core.now_utc().timestamp(), "erasedAt": core.iso(core.now_utc())})
        if consents:   # keep the consent record (proof of withdrawal), minus data
            c.repo.set(user_path(c.uid, "consents"), {"flags": {}, "history": consents.get("history", [])[-100:],
                                                      "erasedAt": core.iso(core.now_utc())})
    else:
        for coll in SCOPES[scope]:
            for doc_id, _ in c.repo.query(user_path(c.uid, coll)):
                n += c.repo.delete_tree(user_path(c.uid, coll, doc_id))
        if scope == "mood":
            c.repo.set(user_path(c.uid, "state"), {"lastMood": None, "moodRev": core.now_utc().timestamp()}, merge=True)
    return {"scope": scope, "deleted": n}


@router.post("/delete-account", summary="Art. 17: delete the account and all data (v1 + legacy + Firebase Auth)")
def delete_account(body: DeleteAccountIn, request: Request, c: Caller = Depends(caller)):
    receipt = {"uid": c.uid, "at": core.iso(core.now_utc()), "v1Deleted": c.repo.delete_tree(f"{ROOT}/{c.uid}")}
    if body.alsoDeleteLegacyData:
        try:
            from compliance_service import get_compliance_service
            receipt["legacy"] = get_compliance_service().delete_all_user_data(c.uid, ip_address=_ip(request)).model_dump(mode="json")
        except Exception as e:
            receipt["legacy"] = {"error": type(e).__name__}
        try:
            from firebase_service import get_firebase_service
            db = get_firebase_service().db
            if db is not None:   # FlutterFlow user document + its subcollections
                db.recursive_delete(db.collection("Users").document(c.uid))
                receipt["flutterflowUserDoc"] = "deleted"
        except Exception as e:
            receipt["flutterflowUserDoc"] = f"error: {type(e).__name__}"
    if body.alsoDeleteAuthAccount:
        try:
            from firebase_admin import auth as fb_auth
            fb_auth.delete_user(c.uid)
            receipt["authAccount"] = "deleted"
        except Exception as e:
            receipt["authAccount"] = f"error: {type(e).__name__}"
    logger.info(f"account deletion completed for {c.uid[:6]}…")
    return receipt
