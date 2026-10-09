"""/v1/selfcare — Self-Care Score and Self-Care Plans."""

from __future__ import annotations

from typing import List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from escape_api import core, selfcare
from escape_api.deps import Caller, caller, has_consent
from escape_api.repo import user_path
from escape_api.settings import cfg

router = APIRouter(prefix="/v1/selfcare", tags=["v1 · self-care"])


class PlanIn(BaseModel):
    goals: List[Literal["sleep_better", "less_stress", "focus", "lift_mood", "move_more", "connect", "know_myself"]] = Field(..., min_length=1, max_length=3)
    minutesPerDay: int = Field(15, ge=5, le=90)
    days: int = Field(7, ge=3, le=14)
    provider: Optional[Literal["catalog", "claude"]] = None


@router.get("/score", summary="Self-Care Score 0-100 with components and one next step (cached per day)")
def score(c: Caller = Depends(caller)):
    return selfcare.compute_score(c.repo, c.uid, c.tz)


@router.get("/score/history", summary="Daily score snapshots")
def score_history(c: Caller = Depends(caller), days: int = Query(30, ge=7, le=180)):
    since = selfcare._since(c.tz, days)
    rows = c.repo.query(user_path(c.uid, "scores"), where=[("day", ">=", since)], order_by="day")
    return {"items": [{k: d.get(k) for k in ("day", "score", "band")} for _, d in rows]}


@router.get("/activities", summary="The self-care activity library (every item is a real in-app destination)")
def activities():
    out = []
    for a in core.dataset("selfcare_activities")["activities"]:
        link = core.resolve_route(a["route"], a.get("params"))
        if link:
            out.append({k: a[k] for k in ("id", "title", "kind", "minutes", "pillar")} | {"deeplink": link})
    return {"items": out}


@router.get("/goals", summary="Goals a plan can be built around")
def goals():
    return {"items": [{"key": k, "label": v["label"]} for k, v in selfcare.GOALS.items()]}


@router.post("/plans", status_code=201, summary="Build a plan (catalog today; 'claude' provider when enabled)")
async def create(body: PlanIn, c: Caller = Depends(caller)):
    prov = body.provider or cfg().plan_provider
    if prov == "claude" and not (has_consent(c, "personalization")
                                 and core.check_quota(c.repo, c.uid, c.tz, "plan_llm", cfg().quota_plan_llm_per_day)):
        prov = "catalog"
    plan = await selfcare.create_plan(c.repo, c.uid, c.tz, list(body.goals), body.minutesPerDay, body.days, prov)
    return {"plan": plan, "today": selfcare.today_items(plan, c.tz)}


@router.get("/plans/active", summary="Active plan + today's steps")
def active(c: Caller = Depends(caller)):
    p = selfcare.active_plan(c.repo, c.uid)
    return {"plan": p, "today": selfcare.today_items(p, c.tz)}


@router.post("/plans/{plan_id}/items/{item_id}/complete", summary="Tick off a plan step (coins + streak)")
def complete_item(plan_id: str, item_id: str, c: Caller = Depends(caller)):
    p = c.repo.get(user_path(c.uid, "plans", plan_id))
    if not p:
        raise HTTPException(404, "Plan not found")
    hit = None
    for d in p["days"]:
        for it in d["items"]:
            if it["itemId"] == item_id:
                hit = it
    if not hit:
        raise HTTPException(404, "Item not found")
    coins = 0
    if not hit["done"]:
        hit["done"], hit["doneAt"] = True, core.iso(core.now_utc())
        c.repo.set(user_path(c.uid, "plans", plan_id), {"days": p["days"]}, merge=True)
        coins = core.award_coins(c.repo, c.uid, c.tz, "plan_item")
        core.touch_activity(c.repo, c.uid, c.tz, "plan")
    return {"item": hit, "coinsAwarded": coins}


@router.delete("/plans/{plan_id}", status_code=204)
def delete_plan(plan_id: str, c: Caller = Depends(caller)):
    c.repo.delete(user_path(c.uid, "plans", plan_id))
