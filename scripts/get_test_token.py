"""
Print a fresh Firebase ID token for the local dev test user.

Reads TEST_USER_EMAIL, TEST_USER_PASSWORD, and FIREBASE_WEB_API_KEY from .env,
calls Firebase's REST API to sign in, and prints the ID token to stdout.

Usage:
    # Get a token
    python scripts/get_test_token.py

    # Use it in curl (bash / git bash / WSL)
    TOKEN=$(python scripts/get_test_token.py)
    curl -H "Authorization: Bearer $TOKEN" http://localhost:8080/chat ...

    # Use it in PowerShell
    $TOKEN = python scripts/get_test_token.py
    curl -H "Authorization: Bearer $TOKEN" http://localhost:8080/chat ...

Tokens expire after 1 hour. Just re-run this script when yours expires.
"""

import base64
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)

import requests


def _wait_for_token_maturity(id_token: str, safety_buffer: float = 1.0) -> None:
    """
    Sleep until the local clock has caught up past the token's `iat` claim.

    Why: Firebase stamps tokens with Google's perfectly-synced server time.
    If the local machine's clock is behind (common on Windows dev boxes that
    haven't run `w32tm /resync` in a while), freshly-issued tokens can look
    "used too early" to the Admin SDK, which rejects them with InvalidIdTokenError.

    This is a dev-only guard — production Cloud Run instances have NTP-synced
    clocks and never hit this.
    """
    try:
        parts = id_token.split(".")
        payload_b64 = parts[1] + "=" * (4 - len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        iat = float(payload.get("iat", 0))
    except Exception:
        return  # malformed token — let the caller discover it
    now = time.time()
    if now < iat + safety_buffer:
        time.sleep(iat + safety_buffer - now)


def main() -> None:
    email = os.getenv("TEST_USER_EMAIL", "").strip()
    password = os.getenv("TEST_USER_PASSWORD", "").strip()
    api_key = os.getenv("FIREBASE_WEB_API_KEY", "").strip()

    missing = [k for k, v in [
        ("TEST_USER_EMAIL", email),
        ("TEST_USER_PASSWORD", password),
        ("FIREBASE_WEB_API_KEY", api_key),
    ] if not v]
    if missing:
        sys.exit(
            f"ERROR: Missing in .env: {', '.join(missing)}. "
            f"Run scripts/create_test_user.py first."
        )

    url = (
        "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword"
        f"?key={api_key}"
    )
    r = requests.post(
        url,
        json={"email": email, "password": password, "returnSecureToken": True},
        timeout=10,
    )
    if r.status_code != 200:
        err = r.json().get("error", {}).get("message", "UNKNOWN")
        sys.exit(f"ERROR: Firebase sign-in failed ({r.status_code}): {err}")

    id_token = r.json().get("idToken")
    if not id_token:
        sys.exit("ERROR: Sign-in succeeded but no idToken in response")

    # Guard against local-clock-behind-real-time: wait until now > iat + 1s
    # so the token is "mature" by the time the caller uses it.
    _wait_for_token_maturity(id_token)

    # Print ONLY the token (no trailing newline noise) so it's easy to
    # capture with TOKEN=$(python scripts/get_test_token.py)
    sys.stdout.write(id_token)


if __name__ == "__main__":
    main()
