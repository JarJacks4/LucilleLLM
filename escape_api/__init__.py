"""
Escape v1 API — the endpoints behind the new Journal, Mood Scan / Mood Stats, Soundscapes AI,
Self-Care Score + Plans, and GDPR controls.

    from escape_api import register
    register(app)        # in main.py, after the app is created

Everything is under /v1 and authenticated with the caller's Firebase ID token
(uid comes from the token, never from the URL). Legacy routes in main.py are untouched.
"""

from fastapi import FastAPI


def register(app: FastAPI) -> None:
    from escape_api.legacy_guard import LegacyAuthGuard
    from escape_api.routers import journal, me, meta, mood, privacy, selfcare, soundscapes

    for r in (meta.router, me.router, mood.router, journal.router, soundscapes.router, selfcare.router, privacy.router):
        app.include_router(r)
    app.add_middleware(LegacyAuthGuard)
