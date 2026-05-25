"""
LucilleLLM - Authentication & Authorization Middleware

Provides Firebase Auth JWT verification and role-based access control.
Supports three auth modes:
  1. Firebase JWT tokens (for mobile/web clients)
  2. API key auth (for service-to-service calls)
  3. No auth (for public endpoints like /health)

Follows the existing middleware pattern from middleware.py.
"""

import logging
import os
from typing import Optional

from fastapi import Depends, HTTPException, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

# Optional security scheme - auto_error=False allows unauthenticated requests
# to pass through to the dependency for proper error handling
security = HTTPBearer(auto_error=False)

# Internal service-to-service auth secret (loaded from environment)
_API_KEY = os.getenv("INTERNAL_SERVICE_KEY", "")


class _AuthError(Exception):
    """Auth verification failed with a specific reason."""
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def _verify_firebase_token(token: str) -> Optional[dict]:
    """
    Verify a Firebase Auth ID token.

    Checks revocation against Firebase Auth (catches logout / disabled accounts).
    Returns the decoded token dict on success.
    Raises _AuthError with a specific reason on failure (revoked, expired, invalid).
    Returns None only when firebase_admin is not installed (degrades gracefully).
    """
    try:
        import firebase_admin.auth as firebase_auth
        # check_revoked=True hits Firebase Auth to verify the token hasn't been
        # revoked (e.g., user logged out or account was disabled). This is one
        # extra round-trip per request but is the correct production setting.
        decoded = firebase_auth.verify_id_token(token, check_revoked=True)
        return decoded
    except ImportError:
        logger.warning("firebase_admin not available for token verification")
        return None
    except Exception as e:
        # Map Firebase exceptions to specific auth errors so callers (and clients)
        # know whether to refresh the token, re-login, or contact support.
        exc_name = type(e).__name__
        if exc_name == "RevokedIdTokenError":
            raise _AuthError(401, "Token has been revoked. Please log in again.")
        if exc_name == "ExpiredIdTokenError":
            raise _AuthError(401, "Token has expired. Please refresh your session.")
        if exc_name == "InvalidIdTokenError":
            raise _AuthError(401, "Invalid authentication token.")
        if exc_name == "UserDisabledError":
            raise _AuthError(403, "Your account has been disabled.")
        # Unknown failure — log full details server-side, return generic message
        logger.debug(f"Firebase token verification failed: {exc_name}: {e}")
        return None


def _verify_api_key(token: str) -> Optional[dict]:
    """
    Verify an API key for service-to-service auth.
    Returns a synthetic user dict or None on failure.
    """
    if _API_KEY and token == _API_KEY:
        return {
            "uid": "service-account",
            "role": "admin",
            "admin": True,  # also expose under the same key Firebase custom claims use
            "auth_method": "api_key",
        }
    return None


def _is_admin(user: dict) -> bool:
    """
    True if the user has admin privileges.

    A user is admin if either:
      1. They authenticated via INTERNAL_SERVICE_KEY (synthetic service account), OR
      2. Their Firebase ID token has the custom claim `admin: true`

    Set the custom claim from a one-off admin script using the Firebase Admin SDK:
        from firebase_admin import auth
        auth.set_custom_user_claims(uid, {"admin": True})
    The user must then refresh their ID token (sign out + sign back in, or call
    getIdToken(true) on the client) before the new claim takes effect.
    """
    if user.get("role") == "admin":  # service-account / API-key path
        return True
    return user.get("admin") is True  # Firebase custom claim


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
) -> dict:
    """
    FastAPI dependency that extracts and validates the authenticated user.

    Tries Firebase JWT first, then INTERNAL_SERVICE_KEY.
    Raises 401 if no valid credentials found.

    Usage:
        @app.get("/protected")
        async def endpoint(user=Depends(get_current_user)):
            user_id = user["uid"]
    """
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Provide a Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    # Try Firebase JWT first. This raises _AuthError with a specific reason
    # (revoked, expired, invalid) on token-shaped failures, or returns None
    # for unknown / non-Firebase tokens so the API key fallback can run.
    try:
        user = _verify_firebase_token(token)
    except _AuthError as e:
        raise HTTPException(
            status_code=e.status_code,
            detail=e.detail,
            headers={"WWW-Authenticate": "Bearer"},
        )
    if user:
        user["auth_method"] = "firebase"
        return user

    # Fallback: internal service key for backend-to-backend / testing
    user = _verify_api_key(token)
    if user:
        return user

    raise HTTPException(
        status_code=401,
        detail="Invalid authentication token.",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def require_admin(user: dict = Depends(get_current_user)) -> dict:
    """
    FastAPI dependency that requires admin privileges.

    Accepts either:
      - INTERNAL_SERVICE_KEY auth (synthetic service account)
      - Firebase user with `admin: true` custom claim

    Usage:
        @app.get("/admin/endpoint", dependencies=[Depends(require_admin)])
        async def admin_endpoint(): ...
    """
    if not _is_admin(user):
        raise HTTPException(
            status_code=403,
            detail="Admin access required.",
        )
    return user


def require_same_user(user_id_param: str = "user_id"):
    """
    Factory that creates a dependency to ensure the authenticated user
    can only access their own data.

    Usage:
        @app.get("/users/{user_id}/data")
        async def get_data(user_id: str, user=Depends(require_same_user())):
            ...
    """
    async def _check(
        request: Request,
        user: dict = Depends(get_current_user),
    ) -> dict:
        path_user_id = request.path_params.get(user_id_param, "")
        # Admins and service accounts can access any user's data
        if _is_admin(user):
            return user
        if user.get("uid") != path_user_id:
            raise HTTPException(
                status_code=403,
                detail="You can only access your own data.",
            )
        return user

    return _check
