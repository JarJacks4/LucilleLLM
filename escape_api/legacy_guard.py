"""
Auth guard for the legacy /users/{user_id}/..., /therapy/{user_id}/..., etc. routes in main.py.

Those routes are KEPT (the current app calls them) but most were written without auth.
  * ALWAYS enforced: the GDPR routes the app never calls (export, data delete, consent).
    Without this, anyone who knows a uid could export or erase that user's data.
  * Enforced for every other user-scoped legacy route when LEGACY_REQUIRE_AUTH=true.
    Turn that on after the FlutterFlow API calls send 'Authorization: Bearer [currentJwtToken]'.
"""

from __future__ import annotations

import os
import re

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

ALWAYS = [
    re.compile(r"^/users/(?P<uid>[^/]+)/export/?$"),
    re.compile(r"^/users/(?P<uid>[^/]+)/data/?$"),
    re.compile(r"^/users/(?P<uid>[^/]+)/consent/?$"),
]
OPTIONAL = [
    re.compile(r"^/users/(?P<uid>(?!onboard$)[^/]+)(/.*)?$"),
    re.compile(r"^/(therapy|feedback|wearables|rl|reviews|assessments)/(?P<uid>(?!exercises$|instruments$|recommend$)[^/]+)(/.*)?$"),
    re.compile(r"^/therapy/recommend/(?P<uid>[^/]+)$"),
    re.compile(r"^/soundscapes/recommend/(?P<uid>[^/]+)$"),
    re.compile(r"^/soundscapes/(?P<uid>[^/]+)/(start|stop|history)(/.*)?$"),
    re.compile(r"^/safety/(?P<uid>[^/]+)/audit$"),
]


def _match(path: str, patterns):
    for p in patterns:
        m = p.match(path)
        if m:
            return m.group("uid")
    return None


class LegacyAuthGuard(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path.startswith("/v1/") or request.method == "OPTIONS":
            return await call_next(request)
        uid = _match(path, ALWAYS)
        if uid is None and os.getenv("LEGACY_REQUIRE_AUTH", "false").lower() in ("1", "true", "yes"):
            uid = _match(path, OPTIONAL)
        if uid is None:
            return await call_next(request)
        auth = request.headers.get("authorization", "")
        if not auth.lower().startswith("bearer "):
            return JSONResponse({"detail": "Authentication required. Provide a Bearer token."}, status_code=401)
        from fastapi.security import HTTPAuthorizationCredentials
        from fastapi import HTTPException
        from auth_middleware import _is_admin, get_current_user
        try:
            user = await get_current_user(HTTPAuthorizationCredentials(scheme="Bearer", credentials=auth[7:].strip()))
        except HTTPException as e:
            return JSONResponse({"detail": e.detail}, status_code=e.status_code)
        if not _is_admin(user) and user.get("uid") != uid:
            return JSONResponse({"detail": "You can only access your own data."}, status_code=403)
        return await call_next(request)
