"""
/v2/plans — Claude-authored self-care plans, their documents and their charts.

    POST   /v2/plans                                  write a plan from real history
    GET    /v2/plans                                  the person's plans (newest first)
    GET    /v2/plans/active                           active plan + today's items
    GET    /v2/plans/{id}                             one plan
    PATCH  /v2/plans/{id}                             easier | harder | shift_days
    POST   /v2/plans/{id}/items/{itemId}/complete     tick a step (coins + score)
    DELETE /v2/plans/{id}                             archive (never hard-delete)
    POST   /v2/plans/{id}/document                    pdf | md | docx -> signed URL
    GET    /v2/plans/{id}/documents/{docId}           download when there is no GCS bucket
    GET    /v2/plans/{id}/chart                       {series, svg} from stored data only

Plans need `personalization` consent. Without it — or without an Anthropic key, or
once the daily Claude quota is spent — the existing catalog provider writes the plan
instead and `provider` says "catalog". The shape never changes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel, Field

from escape_api import core, documents, plans_claude, selfcare
from escape_api.claude import client as claude
from escape_api.deps import Caller, caller, require_consent
from escape_api.settings import cfg

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v2/plans", tags=["v2 · plans (Claude)"])


class PlanIn(BaseModel):
    goals: List[str] = Field(..., min_length=1, max_length=3)
    minutesPerDay: int = Field(15, ge=5, le=90)
    days: int = Field(7, ge=plans_claude.MIN_DAYS, le=plans_claude.MAX_DAYS)
    notes: Optional[str] = Field(None, max_length=1200)


class AdjustIn(BaseModel):
    adjust: Literal["easier", "harder", "shift_days"]
    note: Optional[str] = Field(None, max_length=400)


class DocumentIn(BaseModel):
    format: Literal["pdf", "md", "docx"] = "pdf"
    include: List[Literal["schedule", "progress", "notes"]] = Field(
        default_factory=lambda: ["schedule", "progress", "notes"])


def _plan_or_404(c: Caller, plan_id: str) -> Dict[str, Any]:
    plan = plans_claude.get_plan(c.repo, c.uid, plan_id)
    if not plan:
        raise HTTPException(404, "Plan not found")
    return plan


@router.post("", status_code=201, summary="Write a plan (Claude when available, catalog otherwise)")
async def create(body: PlanIn, c: Caller = Depends(caller)):
    require_consent(c, "personalization")
    use_claude = claude.available() and core.check_quota(c.repo, c.uid, c.tz, "claude_plan",
                                                         cfg().quota_claude_plan_per_day)
    plan, statuses = await plans_claude.author_plan(
        c.repo, c.uid, c.tz, list(body.goals), body.minutesPerDay, body.days, body.notes,
        use_claude=use_claude)
    return {"plan": plan, "today": plans_claude.today_items(plan, c.tz),
            "statuses": statuses,
            "claudeAvailable": claude.available()}


@router.get("", summary="The person's plans, newest first")
def list_plans(c: Caller = Depends(caller), limit: int = Query(20, ge=1, le=50)):
    return {"items": plans_claude.list_plans(c.repo, c.uid, limit=limit)}


@router.get("/active", summary="Active plan + today's items")
def active(c: Caller = Depends(caller)):
    plan = plans_claude.active_plan(c.repo, c.uid)
    return {"plan": plan, "today": plans_claude.today_items(plan, c.tz)}


@router.get("/{plan_id}", summary="One plan")
def get_one(plan_id: str, c: Caller = Depends(caller)):
    return {"plan": _plan_or_404(c, plan_id)}


@router.patch("/{plan_id}", summary="Make it easier, harder, or shift the unfinished days forward")
async def patch_plan(plan_id: str, body: AdjustIn, c: Caller = Depends(caller)):
    require_consent(c, "personalization")
    plan = _plan_or_404(c, plan_id)
    plan, changed = await plans_claude.adjust_plan(c.repo, c.uid, c.tz, plan, body.adjust, body.note)
    return {"plan": plan, "changed": changed, "today": plans_claude.today_items(plan, c.tz)}


@router.post("/{plan_id}/items/{item_id}/complete", summary="Tick off a step (coins + Self-Care Score)")
def complete(plan_id: str, item_id: str, c: Caller = Depends(caller)):
    plan = _plan_or_404(c, plan_id)
    try:
        plan, score, coins = plans_claude.complete_item(c.repo, c.uid, c.tz, plan, item_id)
    except KeyError:
        raise HTTPException(404, "Item not found")
    return {"plan": plan, "score": score, "coins": coins,
            "today": plans_claude.today_items(plan, c.tz)}


@router.delete("/{plan_id}", summary="Archive a plan (never a hard delete)")
def archive(plan_id: str, c: Caller = Depends(caller)):
    _plan_or_404(c, plan_id)
    plan = plans_claude.archive_plan(c.repo, c.uid, plan_id)
    return {"plan": plan, "archived": True}


# ── documents ────────────────────────────────────────────────────────────────

@router.post("/{plan_id}/document", summary="PDF / Markdown / Word of a plan, as a 24 h URL")
def make_document(plan_id: str, body: DocumentIn, c: Caller = Depends(caller)):
    require_consent(c, "personalization")
    plan = _plan_or_404(c, plan_id)
    if not core.check_quota(c.repo, c.uid, c.tz, "claude_doc", cfg().quota_claude_doc_per_day):
        raise HTTPException(429, {"code": "quota_exceeded",
                                  "message": "You've made as many plan documents as today allows."})
    include = list(dict.fromkeys(body.include)) or ["schedule"]
    charts: List[Dict[str, Any]] = []
    progress = None
    if "progress" in include:
        charts = [documents.series(c.repo, c.uid, c.tz, m, "30d", plan) for m in ("adherence", "mood")]
        st = core.get_state(c.repo, c.uid) or {}
        score = selfcare.compute_score(c.repo, c.uid, c.tz)
        progress = {"score": score.get("score"), "band": score.get("band"),
                    "streakDays": int(st.get("streakDays") or 0)}
    try:
        data, mime = documents.render(body.format, plan, include, charts, progress)
    except documents.DocumentUnavailable as e:
        raise HTTPException(503, {"code": e.code, "message": str(e)})
    try:
        rec = documents.store_document(c.repo, c.uid, plan, body.format, data, mime, c.tz)
    except ValueError as e:
        raise HTTPException(507, {"code": "document_too_large", "message": str(e)})
    return {"url": rec["url"], "mime": rec["mime"], "bytes": rec["bytes"],
            "expiresAt": rec["expiresAt"], "id": rec["id"], "filename": rec["filename"],
            "storage": rec["storage"],
            "charts": [{"metric": ch["metric"], "empty": ch["empty"], "note": ch["note"]} for ch in charts]}


@router.get("/{plan_id}/documents/{doc_id}", summary="Download a document kept in Firestore (no GCS bucket)")
def download_document(plan_id: str, doc_id: str, c: Caller = Depends(caller)):
    got = documents.read_document(c.repo, c.uid, doc_id)
    if not got:
        raise HTTPException(404, {"code": "document_expired",
                                  "message": "That document has expired. Make a new one."})
    data, mime, filename = got
    return Response(content=data, media_type=mime,
                    headers={"Content-Disposition": f'attachment; filename="{filename}"',
                             "Cache-Control": "private, max-age=0, no-store"})


# ── charts ───────────────────────────────────────────────────────────────────

@router.get("/{plan_id}/chart", summary="One metric as data + server-rendered SVG (real history only)")
def chart(plan_id: str, c: Caller = Depends(caller),
          metric: Literal["adherence", "minutes", "mood", "score", "streak"] = Query("adherence"),
          range: Literal["7d", "30d", "90d"] = Query("30d")):
    plan = plans_claude.get_plan(c.repo, c.uid, plan_id) if plan_id not in ("none", "active") else \
        plans_claude.active_plan(c.repo, c.uid)
    if plan is None and metric == "adherence":
        raise HTTPException(404, "Plan not found")
    data = documents.series(c.repo, c.uid, c.tz, metric, range, plan)
    return {"series": data, "svg": documents.chart_svg(data),
            "palette": documents.PALETTE}
