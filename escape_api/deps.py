"""FastAPI dependencies for the v1 API: who is calling, their time zone, consent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, Header, HTTPException

from auth_middleware import get_current_user
from escape_api.repo import Repo, get_repo, user_path
from escape_api.settings import cfg

# Consent purposes (GDPR Art. 6/9). Wellbeing data = mood + journal (special category).
PURPOSES = {
    "wellbeing_data": "Store my mood check-ins and journal entries",
    "ai_reflection": "Let Lucille (AI) read an entry to reflect on it",
    "camera_scan": "Use my camera for the pulse part of the Mood Scan (frames stay on the phone)",
    "voice": "Use my microphone for voice journaling and voice mood input",
    "health_data": "Read Apple Health / Health Connect data (heart rate, sleep)",
    "location_weather": "Use my approximate location for Inner Weather",
    "personalization": "Use my history to personalise suggestions and soundscapes",
    "analytics": "Share anonymous usage analytics to improve Escape",
}


@dataclass
class Caller:
    uid: str
    tz: str
    repo: Repo
    is_admin: bool = False


async def caller(
    user: dict = Depends(get_current_user),
    x_timezone: Optional[str] = Header(default=None, alias="X-Timezone"),
) -> Caller:
    uid = user.get("uid")
    if not uid:
        raise HTTPException(401, "Authentication required.")
    repo = get_repo()
    tz = x_timezone
    if not tz:
        st = repo.get(user_path(uid, "state")) or {}
        tz = st.get("timezone") or "UTC"
    return Caller(uid=uid, tz=tz, repo=repo,
                  is_admin=user.get("role") == "admin" or user.get("admin") is True)


def consents(c: Caller) -> dict:
    return (c.repo.get(user_path(c.uid, "consents")) or {}).get("flags", {})


def require_consent(c: Caller, purpose: str) -> None:
    """Raise 403 consent_required unless the user opted in (or enforcement is off)."""
    if not cfg().enforce_consent or c.is_admin:
        return
    if not consents(c).get(purpose):
        raise HTTPException(
            status_code=403,
            detail={"code": "consent_required", "purpose": purpose, "label": PURPOSES.get(purpose, purpose),
                    "action": "PUT /v1/privacy/consents"},
        )


def has_consent(c: Caller, purpose: str) -> bool:
    if not cfg().enforce_consent:
        return True
    return bool(consents(c).get(purpose))
