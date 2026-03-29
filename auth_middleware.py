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

# API key for service-to-service auth (loaded from environment)
_API_KEY = os.getenv("LUCILLE_API_KEY", "")


def _verify_firebase_token(token: str) -> Optional[dict]:
    """
    Verify a Firebase Auth ID token.
    Returns the decoded token dict or None on failure.
    """
    try:
        import firebase_admin.auth as firebase_auth
        decoded = firebase_auth.verify_id_token(token)
        return decoded
    except ImportError:
        logger.warning("firebase_admin not available for token verification")
        return None
    except Exception as e:
        logger.debug(f"Firebase token verification failed: {e}")
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
            "auth_method": "api_key",
        }
    return None


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
) -> dict:
    """
    FastAPI dependency that extracts and validates the authenticated user.

    Tries Firebase JWT first, then API key.
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

    # Try Firebase JWT
    user = _verify_firebase_token(token)
    if user:
        user["auth_method"] = "firebase"
        return user

    # Try API key
    user = _verify_api_key(token)
    if user:
        return user

    raise HTTPException(
        status_code=401,
        detail="Invalid or expired token.",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def require_admin(user: dict = Depends(get_current_user)) -> dict:
    """
    FastAPI dependency that requires admin role.

    Usage:
        @app.get("/admin/endpoint", dependencies=[Depends(require_admin)])
        async def admin_endpoint(): ...
    """
    role = user.get("role", "")
    if role != "admin":
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
        if user.get("role") == "admin":
            return user
        if user.get("uid") != path_user_id:
            raise HTTPException(
                status_code=403,
                detail="You can only access your own data.",
            )
        return user

    return _check
