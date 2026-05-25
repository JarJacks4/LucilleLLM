"""
Create (or rotate) a Firebase test user for local development.

What it does:
  1. Initializes Firebase Admin using FIREBASE_CREDENTIAL_PATH from .env
  2. Creates a test user with email test@lucille.local + a strong random password
     (if the user already exists, rotates its password instead of erroring)
  3. Grants the user the `admin: true` custom claim so it can hit /admin/* endpoints
  4. Verifies sign-in works by calling Firebase's REST API (signInWithPassword)
     using FIREBASE_WEB_API_KEY — this catches the "Email/Password provider not
     enabled in Firebase Console" error early
  5. Prints a summary (UID only — never prints the password to stdout)

After this runs, the test user credentials are in .env as:
    TEST_USER_EMAIL=...
    TEST_USER_PASSWORD=...
    TEST_USER_UID=...

Usage:
    python scripts/create_test_user.py

Idempotent: safe to re-run. Each run rotates the password.
"""

import os
import secrets
import sys
from pathlib import Path

# Make project root importable and load .env
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)

import firebase_admin
from firebase_admin import credentials, auth
import requests


TEST_EMAIL = "test@lucille.local"


def _init_firebase() -> None:
    """Initialize the Admin SDK using the local service account file."""
    if firebase_admin._apps:
        return
    cred_path = os.getenv("FIREBASE_CREDENTIAL_PATH", "").strip()
    if not cred_path or not (ROOT / cred_path).exists():
        sys.exit(
            f"ERROR: FIREBASE_CREDENTIAL_PATH not set or file not found: {cred_path!r}"
        )
    cred = credentials.Certificate(str(ROOT / cred_path))
    firebase_admin.initialize_app(cred)


def _generate_password() -> str:
    """Strong random password — URL-safe so it's easy to pass in shell env vars."""
    return secrets.token_urlsafe(24)


def _create_or_rotate_user(email: str, password: str) -> str:
    """
    Create the test user, or if they already exist, rotate their password.
    Returns the user's UID.
    """
    try:
        user = auth.create_user(
            email=email,
            password=password,
            email_verified=True,
            display_name="Lucille Local Dev Test User",
        )
        print(f"OK  created new user: {user.uid}")
    except auth.EmailAlreadyExistsError:
        user = auth.get_user_by_email(email)
        auth.update_user(user.uid, password=password)
        print(f"OK  rotated password on existing user: {user.uid}")
    # Set admin custom claim so the user can hit /admin/* endpoints
    auth.set_custom_user_claims(user.uid, {"admin": True})
    print(f"OK  granted admin custom claim")
    return user.uid


def _verify_signin(email: str, password: str) -> None:
    """
    Try to sign in via Firebase REST API. Catches the common
    'Email/Password provider not enabled' error early.
    """
    api_key = os.getenv("FIREBASE_WEB_API_KEY", "").strip()
    if not api_key:
        print("SKIP verify: FIREBASE_WEB_API_KEY not set in .env")
        return
    url = (
        "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword"
        f"?key={api_key}"
    )
    r = requests.post(
        url,
        json={"email": email, "password": password, "returnSecureToken": True},
        timeout=10,
    )
    if r.status_code == 200:
        id_token = r.json().get("idToken", "")
        print(f"OK  sign-in verified, got ID token ({len(id_token)} chars)")
        return
    err = r.json().get("error", {}).get("message", "UNKNOWN")
    if err == "OPERATION_NOT_ALLOWED":
        print()
        print("=" * 70)
        print("ACTION REQUIRED: Enable Email/Password provider in Firebase Console")
        print("=" * 70)
        print("The user was created, but sign-in is blocked because")
        print("Email/Password authentication is disabled for this project.")
        print()
        print("Fix (30 seconds):")
        print("  1. Open https://console.firebase.google.com/project/"
              "escape-self-care-ai/authentication/providers")
        print("  2. Click 'Email/Password' in the list")
        print("  3. Toggle 'Enable' to ON (first toggle, not passwordless)")
        print("  4. Click 'Save'")
        print("  5. Re-run: python scripts/create_test_user.py")
        print("=" * 70)
        sys.exit(2)
    sys.exit(f"FAIL sign-in verification: {err}")


def _update_env_file(email: str, password: str, uid: str) -> None:
    """
    Append or update the TEST_USER_* lines in .env without touching other lines.
    """
    env_path = ROOT / ".env"
    lines = env_path.read_text().splitlines() if env_path.exists() else []

    new_values = {
        "TEST_USER_EMAIL": email,
        "TEST_USER_PASSWORD": password,
        "TEST_USER_UID": uid,
    }

    # Remove any existing TEST_USER_* lines
    filtered = [
        line for line in lines
        if not any(line.startswith(f"{k}=") for k in new_values)
    ]
    # Remove any trailing blank lines so we append cleanly
    while filtered and not filtered[-1].strip():
        filtered.pop()

    filtered.append("")
    filtered.append("# Local dev test user (created by scripts/create_test_user.py)")
    for k, v in new_values.items():
        filtered.append(f"{k}={v}")

    env_path.write_text("\n".join(filtered) + "\n")
    print(f"OK  wrote TEST_USER_* to {env_path}")


def main() -> None:
    _init_firebase()
    password = _generate_password()
    uid = _create_or_rotate_user(TEST_EMAIL, password)
    _verify_signin(TEST_EMAIL, password)
    _update_env_file(TEST_EMAIL, password, uid)
    print()
    print("Done. Credentials are in .env (TEST_USER_EMAIL / PASSWORD / UID).")
    print(f"UID: {uid}")


if __name__ == "__main__":
    main()
