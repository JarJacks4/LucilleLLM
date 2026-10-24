"""Session completion endpoint with idempotent rewards."""
from typing import Literal, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from escape_api import core
from escape_api.deps import Caller, caller
from escape_api.repo import user_path

router = APIRouter(prefix="/v1/sessions", tags=["v1 · sessions"])

class SessionCompleteIn(BaseModel):
    area: Literal["mind", "body", "journal", "sound"]
    sessionId: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")
    title: Optional[str] = Field(default=None, max_length=160)
    minutes: float = Field(ge=0, le=600)
    moodBefore: Optional[str] = Field(default=None, max_length=80)

@router.post("/complete", status_code=201)
def complete_session(body: SessionCompleteIn, c: Caller = Depends(caller)):
    path = user_path(c.uid, "session_completions", body.sessionId)
    old = c.repo.get(path)
    if old:
        state = core.get_state(c.repo, c.uid)
        return {**old, "duplicate": True, "coinsAwarded": 0, "dailyBonus": 0,
                "coins": state["coins"], "streakDays": state["streakDays"]}
    streak = core.touch_activity(c.repo, c.uid, c.tz, body.area)["streakDays"]
    base = core.award_coins(c.repo, c.uid, c.tz, "session_complete") if body.minutes >= 1 else 0
    bonus = core.award_coins(c.repo, c.uid, c.tz, "session_daily_bonus") if base else 0
    state = core.get_state(c.repo, c.uid)
    record = {"area": body.area, "sessionId": body.sessionId, "title": body.title,
              "minutes": body.minutes, "coinsAwarded": base + bonus,
              "dailyBonus": bonus, "streakDays": streak, "coins": state["coins"],
              "duplicate": False, "completedAt": core.iso(core.now_utc())}
    c.repo.set(path, record)
    return record
