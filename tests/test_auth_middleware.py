"""
Tests for auth_middleware: RBAC, INTERNAL_SERVICE_KEY, token revocation.

These tests cover the Phase 2 auth hardening:
  - Custom-claim admin detection (Firebase 'admin: true')
  - Service-account fallback via INTERNAL_SERVICE_KEY
  - Specific JWT failure modes (revoked, expired, invalid, disabled)
  - require_admin / require_same_user accept both auth modes
"""

import os
import pytest
from unittest.mock import patch
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials


# ── Fixtures ──────────────────────────────────────────────


@pytest.fixture
def service_key(monkeypatch):
    """Set a known INTERNAL_SERVICE_KEY for the test, then reload the module."""
    test_key = "test-service-key-12345"
    monkeypatch.setenv("INTERNAL_SERVICE_KEY", test_key)
    # auth_middleware reads the env var at import time, so reload it
    import importlib
    import auth_middleware
    importlib.reload(auth_middleware)
    yield test_key
    # cleanup: reload again after env var is removed by monkeypatch
    importlib.reload(auth_middleware)


def _bearer(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


# ── _is_admin ─────────────────────────────────────────────


def test_is_admin_recognizes_firebase_custom_claim():
    from auth_middleware import _is_admin
    assert _is_admin({"uid": "user123", "admin": True}) is True


def test_is_admin_recognizes_service_account_role():
    from auth_middleware import _is_admin
    assert _is_admin({"uid": "service-account", "role": "admin"}) is True


def test_is_admin_rejects_regular_user():
    from auth_middleware import _is_admin
    assert _is_admin({"uid": "user123"}) is False
    assert _is_admin({"uid": "user123", "admin": False}) is False
    assert _is_admin({"uid": "user123", "role": "user"}) is False


def test_is_admin_rejects_truthy_non_true_admin_value():
    """admin must be exactly True, not 'true' or 1, to prevent claim confusion."""
    from auth_middleware import _is_admin
    assert _is_admin({"uid": "user", "admin": "true"}) is False
    assert _is_admin({"uid": "user", "admin": 1}) is False


# ── _verify_api_key ───────────────────────────────────────


def test_verify_api_key_accepts_correct_key(service_key):
    from auth_middleware import _verify_api_key
    user = _verify_api_key(service_key)
    assert user is not None
    assert user["uid"] == "service-account"
    assert user["role"] == "admin"
    assert user["admin"] is True
    assert user["auth_method"] == "api_key"


def test_verify_api_key_rejects_wrong_key(service_key):
    from auth_middleware import _verify_api_key
    assert _verify_api_key("wrong-key") is None


def test_verify_api_key_rejects_empty_when_unset(monkeypatch):
    """If INTERNAL_SERVICE_KEY isn't set, no token should ever match."""
    monkeypatch.delenv("INTERNAL_SERVICE_KEY", raising=False)
    import importlib, auth_middleware
    importlib.reload(auth_middleware)
    assert auth_middleware._verify_api_key("") is None
    assert auth_middleware._verify_api_key("anything") is None


# ── get_current_user end-to-end ───────────────────────────


@pytest.mark.asyncio
async def test_get_current_user_no_credentials_returns_401():
    from auth_middleware import get_current_user
    with pytest.raises(HTTPException) as exc:
        await get_current_user(credentials=None)
    assert exc.value.status_code == 401
    assert "Authentication required" in exc.value.detail


@pytest.mark.asyncio
async def test_get_current_user_accepts_service_key(service_key):
    from auth_middleware import get_current_user
    user = await get_current_user(credentials=_bearer(service_key))
    assert user["uid"] == "service-account"
    assert user["admin"] is True


@pytest.mark.asyncio
async def test_get_current_user_invalid_token_returns_401(service_key):
    """An unknown token, when Firebase verification fails silently, should be 401."""
    from auth_middleware import get_current_user
    with patch("auth_middleware._verify_firebase_token", return_value=None):
        with pytest.raises(HTTPException) as exc:
            await get_current_user(credentials=_bearer("garbage-token"))
        assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_get_current_user_revoked_token_surfaces_specific_error():
    """A revoked Firebase token should produce a clear 401 with the right detail."""
    from auth_middleware import get_current_user, _AuthError
    with patch(
        "auth_middleware._verify_firebase_token",
        side_effect=_AuthError(401, "Token has been revoked. Please log in again."),
    ):
        with pytest.raises(HTTPException) as exc:
            await get_current_user(credentials=_bearer("any-token"))
        assert exc.value.status_code == 401
        assert "revoked" in exc.value.detail.lower()


@pytest.mark.asyncio
async def test_get_current_user_expired_token_surfaces_specific_error():
    from auth_middleware import get_current_user, _AuthError
    with patch(
        "auth_middleware._verify_firebase_token",
        side_effect=_AuthError(401, "Token has expired. Please refresh your session."),
    ):
        with pytest.raises(HTTPException) as exc:
            await get_current_user(credentials=_bearer("any-token"))
        assert exc.value.status_code == 401
        assert "expired" in exc.value.detail.lower()


@pytest.mark.asyncio
async def test_get_current_user_disabled_account_returns_403():
    from auth_middleware import get_current_user, _AuthError
    with patch(
        "auth_middleware._verify_firebase_token",
        side_effect=_AuthError(403, "Your account has been disabled."),
    ):
        with pytest.raises(HTTPException) as exc:
            await get_current_user(credentials=_bearer("any-token"))
        assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_get_current_user_firebase_user_with_admin_claim():
    """A real Firebase user whose token has the admin custom claim is treated as admin."""
    from auth_middleware import get_current_user, _is_admin
    fake_decoded = {"uid": "real-user-uid", "admin": True, "email": "x@y.com"}
    with patch("auth_middleware._verify_firebase_token", return_value=fake_decoded):
        user = await get_current_user(credentials=_bearer("valid-token"))
    assert user["uid"] == "real-user-uid"
    assert user["auth_method"] == "firebase"
    assert _is_admin(user) is True


# ── require_admin ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_require_admin_accepts_service_account():
    from auth_middleware import require_admin
    user = {"uid": "service-account", "role": "admin", "admin": True}
    assert await require_admin(user=user) == user


@pytest.mark.asyncio
async def test_require_admin_accepts_firebase_admin_claim():
    from auth_middleware import require_admin
    user = {"uid": "real-user", "admin": True}
    assert await require_admin(user=user) == user


@pytest.mark.asyncio
async def test_require_admin_rejects_regular_user():
    from auth_middleware import require_admin
    user = {"uid": "regular-user"}
    with pytest.raises(HTTPException) as exc:
        await require_admin(user=user)
    assert exc.value.status_code == 403
    assert "admin" in exc.value.detail.lower()
