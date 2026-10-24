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
    from escape_api.routers import journal, me, meta, mood, privacy, selfcare, sessions, soundscapes, voice

    for r in (meta.router, me.router, mood.router, journal.router, soundscapes.router, selfcare.router, privacy.router, sessions.router, voice.router):
        app.include_router(r)
    register_claude(app)
    app.add_middleware(LegacyAuthGuard)


def register_claude(app: FastAPI) -> bool:
    """
    Mount the Claude edition (/v2/lucille/*, /v2/plans/*) when LUCILLE_EDITION=claude.

    Default is 'openai', so the app that ships today is untouched: the v2 routes do
    not exist at all. The flag is reported by GET /v1/config as features.lucilleEdition.
    """
    from escape_api.settings import cfg
    if cfg().lucille_edition != "claude":
        return False
    from escape_api.routers import lucille_claude, plans_v2
    for r in (lucille_claude.router, plans_v2.router):
        app.include_router(r)
    return True
